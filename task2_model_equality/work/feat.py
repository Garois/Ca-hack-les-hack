"""Feature extraction for model-equality tests (one family per call, one GPU).

For every response token x_t of every endpoint in the family, with the local base model logits l_t:
  pass 1: log-likelihood of each response on a temperature grid  -> per-endpoint MLE temperature T_e
  pass 2 (at T_e, p_t = softmax(l_t / T_e)):
    * residual vector     R_r = sum_t onehot(x_t) - p_t                   (V-dim, per response; global logit-bias score)
    * hidden residual     g_t = W^T (onehot(x_t) - p_t)                    (d-dim, per token)  + hidden state h_t
    * quantisation scores for each local quantised copy q (d_t = l^q_t - l_t):
          S_q = sum_t (d_t[x_t] - E_p d_t) / T_e     (centered score, mean 0 under the base model)
          V_q = sum_t Var_p(d_t) / T_e^2              (its variance under the base model)
          L_q = sum_t log p(x_t) - log p^q(x_t)       (raw LLR at T=1, as in llr.py)
All sums are kept per response so that statistics can be bootstrapped / sign-flipped over responses.

Usage: python feat.py TAG MODEL_ID DATA [DATA ...] [--eps ep-01,ep-02,...] [--schemes fp8_tensor,int8_channel]
Writes data/feat_<TAG>.pt
"""
import argparse
import copy
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import common
from extq import EXTERNAL, apply_external
from llr import quantize_
from score import build

DEV = "cuda"
TGRID = torch.round(torch.arange(0.40, 1.5001, 0.02) * 100) / 100


EOS_FIX = True


def encode(tok, rows):
    """context ids (default chat template) and response ids; a response shorter than 250 tokens ended naturally
    (max_tokens = 256), so the end-of-turn token the server sampled is appended."""
    out = []
    for r in rows:
        c = tok(build(tok, r["prompt"], None), add_special_tokens=False)["input_ids"]
        x = tok(r["response"], add_special_tokens=False)["input_ids"]
        if EOS_FIX and len(x) < 250:
            x = x + [tok.eos_token_id]
        if len(x) >= 2:
            out.append((r, c, x))
    return out


@torch.no_grad()
def fwd(model, c, x):
    ids = torch.tensor([c + x], device=DEV)
    out = model(ids, output_hidden_states=True)
    return out.logits[0, len(c) - 1:-1].float(), out.hidden_states[-1][0, len(c) - 1:-1].float(), ids[0, len(c):]


def act_quant_hook(mod, args):
    """dynamic per-tensor fp8 activation quantisation (vLLM W8A8 fp8 style)"""
    a = args[0]
    s = a.abs().amax().clamp(min=1e-12) / 448.0
    return ((a.float() / s).to(torch.float8_e4m3fn).float() * s).to(a.dtype),


LOCAL = ["fp8_tensor", "fp8_w8a8", "fp8_channel", "int8_channel", "int4_g128", "nf4_b64"]


@torch.no_grad()
def quant_scores(model, model_id, enc, Tfit, scheme):
    qm = copy.deepcopy(model)
    hooks = []
    if scheme in EXTERNAL.get(model_id, {}):
        hooks = apply_external(qm, EXTERNAL[model_id][scheme], scheme)
    elif scheme == "fp8_w8a8":
        quantize_(qm, "fp8_tensor")
        for name, mod in qm.named_modules():
            if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
                hooks.append(mod.register_forward_pre_hook(act_quant_hook))
    else:
        quantize_(qm, scheme)
    S = torch.zeros(len(enc)); Vq = torch.zeros(len(enc)); L = torch.zeros(len(enc)); n = torch.zeros(len(enc))
    kls = []
    for i, (r, c, x) in enumerate(enc):
        lg, _, xt = fwd(model, c, x)
        lq, _, _ = fwd(qm, c, x)
        T = Tfit[r["endpoint"]]
        p = torch.softmax(lg / T, -1)
        d = lq - lg
        Ed = (p * d).sum(-1)
        S[i] = ((d.gather(1, xt[:, None])[:, 0] - Ed) / T).sum().cpu()
        Vq[i] = (((p * d * d).sum(-1) - Ed ** 2) / T ** 2).sum().cpu()
        L[i] = (torch.log_softmax(lg, -1).gather(1, xt[:, None])[:, 0]
                - torch.log_softmax(lq, -1).gather(1, xt[:, None])[:, 0]).sum().cpu()
        n[i] = len(xt)
        if i < 10:
            kls.append((torch.softmax(lg, -1) * (torch.log_softmax(lg, -1) - torch.log_softmax(lq, -1))).sum(-1).mean().item())
    print(f"  [{scheme}] mean KL(base||quant) per token on 10 responses: {np.mean(kls):.4f}", flush=True)
    for hk in hooks:
        hk.remove()
    del qm
    torch.cuda.empty_cache()
    return {"S": S, "V": Vq, "L": L, "n": n}


def report(q, scheme, ep_of, eps, t0):
    zs = []
    for e in eps:
        m = torch.tensor(ep_of == e)
        zs.append(f"{e}:{(q['S'][m].sum() / q['V'][m].sum().sqrt()).item():+.1f}")
    print(f"[{scheme}] z(theory) " + " ".join(zs), f"({time.time() - t0:.0f}s)", flush=True)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("model")
    ap.add_argument("data", nargs="+")
    ap.add_argument("--eps", default=None)
    ap.add_argument("--schemes", default="fp8_tensor,fp8_w8a8,fp8_channel,int8_channel,int4_g128,nf4_b64,awq,gptq4,gptq8,ctint8,ctfp8")
    ap.add_argument("--nprev", type=int, default=200)
    ap.add_argument("--ktok", type=int, default=1500)
    ap.add_argument("--quant-only", default=None, help="feat tag whose Tfit is reused; saves data/qx_<tag>.pt")
    a = ap.parse_args()
    keep = set(a.eps.split(",")) if a.eps else None
    rows = [r for d in a.data for r in common.load(d) if keep is None or r["endpoint"] in keep]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, device_map=DEV).eval()
    W = model.get_output_embeddings().weight.detach().float()
    enc = encode(tok, rows)
    eps = sorted({r["endpoint"] for r, _, _ in enc})
    print(f"{a.tag}: {len(enc)} responses, {len(eps)} endpoints, model {a.model}", flush=True)
    t0 = time.time()
    if a.quant_only:
        F = torch.load(common.DATA / f"feat_{a.quant_only}.pt", weights_only=False)
        Tfit = F["Tfit"]
        enc = [(r, c, x) for r, c, x in enc if r["endpoint"] in Tfit]
        ep_of = np.array([r["endpoint"] for r, _, _ in enc])
        assert [m["id"] for m in F["meta"]] == [r["id"] for r, _, _ in enc], "data mismatch with feat file"
        quant = {}
        for scheme in [s for s in a.schemes.split(",") if s]:
            if scheme in EXTERNAL.get(a.model, {}) or scheme in LOCAL:
                quant[scheme] = quant_scores(model, a.model, enc, Tfit, scheme)
                report(quant[scheme], scheme, ep_of, eps, t0)
        torch.save(quant, common.DATA / f"qx_{a.tag}.pt")
        return

    # ---------- pass 1: temperature grid ----------
    G = TGRID.to(DEV)
    llg = torch.zeros(len(enc), len(G))
    for i, (r, c, x) in enumerate(enc):
        lg, _, xt = fwd(model, c, x)
        lx = lg.gather(1, xt[:, None])[:, 0]
        for j, T in enumerate(G):
            llg[i, j] = (lx / T - torch.logsumexp(lg / T, -1)).sum().cpu()
    ep_of = np.array([r["endpoint"] for r, _, _ in enc])
    Tfit = {}
    for e in eps:
        m = torch.tensor(ep_of == e)
        ll = llg[m].sum(0).numpy()
        j = int(ll.argmax())
        T = float(TGRID[j])
        if 0 < j < len(ll) - 1:  # parabolic refinement
            y0, y1, y2 = ll[j - 1], ll[j], ll[j + 1]
            den = y0 - 2 * y1 + y2
            if den < 0:
                T += 0.02 * 0.5 * (y0 - y2) / den
        Tfit[e] = T
    print("T:", {e: round(t, 3) for e, t in Tfit.items()}, f"({time.time() - t0:.0f}s)", flush=True)

    # ---------- pass 2: residuals at T_e ----------
    V = W.shape[0]
    from collections import Counter
    cnt_prev, cnt_tok = Counter(), Counter()
    for r, c, x in enc:
        cnt_prev.update([c[-1]] + x[:-1]); cnt_tok.update(x)
    Pset = [t for t, _ in cnt_prev.most_common(a.nprev)]
    Kset = [t for t, _ in cnt_tok.most_common(a.ktok)]
    pmap = torch.full((V,), -1, dtype=torch.long, device=DEV); pmap[torch.tensor(Pset, device=DEV)] = torch.arange(len(Pset), device=DEV)
    Kidx = torch.tensor(Kset, device=DEV)
    Rb = torch.zeros(len(enc), len(Pset), len(Kset), dtype=torch.float16)
    Rv = torch.zeros(len(enc), V, dtype=torch.float16)
    H, Gr, X, RID = [], [], [], []
    tokstats = []
    for i, (r, c, x) in enumerate(enc):
        lg, h, xt = fwd(model, c, x)
        T = Tfit[r["endpoint"]]
        p = torch.softmax(lg / T, -1)
        oh = torch.zeros_like(p).scatter_(1, xt[:, None], 1.0)
        res = oh - p
        Rv[i] = res.sum(0).half().cpu()
        prev = torch.cat([torch.tensor([c[-1]], device=DEV), xt[:-1]])
        pi = pmap[prev]
        sel = pi >= 0
        acc = torch.zeros(len(Pset), len(Kset), device=DEV)
        acc.index_add_(0, pi[sel], res[sel][:, Kidx])
        Rb[i] = acc.half().cpu()
        g = W[xt] - p @ W
        H.append(h.half().cpu()); Gr.append(g.half().cpu()); X.append(xt.cpu())
        RID.append(torch.full((len(xt),), i, dtype=torch.int32))
        lpT = torch.log(p.gather(1, xt[:, None])[:, 0].clamp(min=1e-30))
        ent = -(p * torch.log(p.clamp(min=1e-30))).sum(-1)
        rank = (p > p.gather(1, xt[:, None])).sum(1)
        tokstats.append(torch.stack([lpT, ent, rank.float()], 1).cpu())
    print(f"pass2 done ({time.time() - t0:.0f}s)", flush=True)

    # ---------- pass 3: quantisation scores ----------
    quant = {}
    for scheme in [s for s in a.schemes.split(",") if s]:
        if scheme in EXTERNAL.get(a.model, {}) or scheme in LOCAL:
            quant[scheme] = quant_scores(model, a.model, enc, Tfit, scheme)
            report(quant[scheme], scheme, ep_of, eps, t0)
    meta = [{"endpoint": r["endpoint"], "id": r["id"], "repeat": r["repeat"], "ntok": len(x)} for r, _, x in enc]
    torch.save({"model": a.model, "meta": meta, "eps": eps, "Tfit": Tfit, "TGRID": TGRID, "llg": llg, "Rv": Rv,
                "H": torch.cat(H), "G": torch.cat(Gr), "X": torch.cat(X), "RID": torch.cat(RID),
                "tok": torch.cat(tokstats), "quant": quant,
                "Rb": Rb, "Pset": Pset, "Kset": Kset}, common.DATA / f"feat_{a.tag}.pt")
    print(f"saved feat_{a.tag}.pt ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
