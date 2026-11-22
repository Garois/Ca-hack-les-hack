"""Learned-direction test (LDT).

Model of an endpoint's next-token distribution, using the local base model's final hidden state h_t:
    logits_t = W (h_t + A B^T h_t) / T          (W = lm_head, A,B: d x r, low-rank "adapter" on the last hidden state)
H0 (same weights as base, any temperature):  A B^T = 0.
We fit (T, A, B) by max likelihood on a train split of the endpoint's responses and measure the log-likelihood gain over
the T-only model on held-out responses (K-fold over responses). Gain > 0 consistently => endpoint deviates from base.

Also: transfer. Fit the direction D = A B^T on endpoint E (all data), then for every endpoint F fit (T, alpha):
    logits = W (h + alpha * D h) / T
alpha_F ~ 0 for base, ~1 for E itself / same model, other values for other strengths of the same adapter.

Usage:
  python ldt.py extract --fam Q15 DATA [DATA ...]         # caches hidden states to data/hid_<fam>.pt
  python ldt.py extract --model MODEL_ID --tag ref DATA [DATA ...]
  python ldt.py cv TAG [--rank 4]                          # per-endpoint held-out gain
  python ldt.py transfer TAG [--rank 4]                    # alpha matrix
"""
import argparse
import math

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import common
from score import build

DEV = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@torch.no_grad()
def extract(mid, rows, tag):
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32, device_map=DEV).eval()
    W = model.get_output_embeddings().weight.detach().float().cpu()
    store = {}
    for r in rows:
        ids_ctx = tok(build(tok, r["prompt"], None), add_special_tokens=False)["input_ids"]
        ids_resp = tok(r["response"], add_special_tokens=False)["input_ids"]
        if len(ids_resp) < 2:
            continue
        ids = torch.tensor([ids_ctx + ids_resp], device=DEV)
        out = model(ids, output_hidden_states=True)
        h = out.hidden_states[-1][0, len(ids_ctx) - 1:-1].float()
        # sanity: W h should reproduce logits
        lg = out.logits[0, len(ids_ctx) - 1:-1].float()
        err = (h @ W.to(DEV).T - lg).abs().max().item()
        if err > 1e-2:
            raise RuntimeError(f"hidden state is not the lm_head input (max err {err})")
        s = store.setdefault(r["endpoint"], {"h": [], "x": [], "rid": []})
        s["h"].append(h.half().cpu())
        s["x"].append(ids[0, len(ids_ctx):].cpu())
        s["rid"].append(torch.full((len(ids_resp),), len(s["rid"]), dtype=torch.long))
    data = {ep: {k: torch.cat(v[k]) for k in v} for ep, v in store.items()}
    torch.save({"W": W.half(), "data": data, "model": mid}, common.DATA / f"hid_{tag}.pt")
    for ep in sorted(data):
        print(ep, data[ep]["h"].shape[0], "tokens", int(data[ep]["rid"].max()) + 1, "responses")


def nll(W, h, x, logT, A=None, Z=None, alpha=None, D=None, bs=4096):
    """mean NLL over tokens (differentiable).
    logits = (W h + W A z) / T            with z = Z[i] (projected, standardized hidden features) when A is given
           = (W h + alpha * W D z) / T    when a fixed direction D (d x r) is given."""
    tot = 0.0
    for i in range(0, len(x), bs):
        hh = h[i:i + bs]
        if A is not None:
            hh = hh + Z[i:i + bs] @ A.T
        if D is not None:
            hh = hh + alpha * (Z[i:i + bs] @ D.T)
        lg = (hh @ W.T) * torch.exp(-logT)
        tot = tot + torch.nn.functional.cross_entropy(lg, x[i:i + bs], reduction="sum")
    return tot / len(x)


def fit_T(W, h, x, steps=60):
    logT = torch.tensor(math.log(0.7), device=DEV, requires_grad=True)
    opt = torch.optim.Adam([logT], lr=0.05)
    for _ in range(steps):
        opt.zero_grad(); loss = nll(W, h, x, logT); loss.backward(); opt.step()
    return logT.detach()


def fit_dir(W, h, x, Z, rid, lr=3e-3, max_steps=300, every=10, val_frac=0.2, seed=0):
    """Learn A (d x r) with temperature, early-stopped on an inner validation split of responses."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ur = rid.unique()
    perm = ur[torch.randperm(len(ur), generator=g).to(ur.device)]
    nval = max(1, int(len(ur) * val_frac))
    isval = torch.isin(rid, perm[:nval])
    tr, va = ~isval, isval
    logT = fit_T(W, h[tr], x[tr]).clone().requires_grad_(True)
    A = torch.zeros(h.shape[1], Z.shape[1], device=DEV, requires_grad=True)
    opt = torch.optim.Adam([logT, A], lr=lr)
    with torch.no_grad():
        best = nll(W, h[va], x[va], logT, A, Z[va]).item()
    best_state = (logT.detach().clone(), A.detach().clone(), 0)
    for step in range(1, max_steps + 1):
        opt.zero_grad()
        loss = nll(W, h[tr], x[tr], logT, A, Z[tr])
        loss.backward()
        opt.step()
        if step % every == 0:
            with torch.no_grad():
                v = nll(W, h[va], x[va], logT, A, Z[va]).item()
            if v < best:
                best, best_state = v, (logT.detach().clone(), A.detach().clone(), step)
    return {"logT": best_state[0], "A": best_state[1], "steps": best_state[2]}


def fit_alpha(W, h, x, Z, D, steps=80):
    logT = torch.tensor(math.log(0.7), device=DEV, requires_grad=True)
    alpha = torch.zeros((), device=DEV, requires_grad=True)
    opt = torch.optim.Adam([logT, alpha], lr=0.05)
    for _ in range(steps):
        opt.zero_grad(); loss = nll(W, h, x, logT, None, Z, alpha, D); loss.backward(); opt.step()
    with torch.no_grad():
        l1 = nll(W, h, x, logT, None, Z, alpha, D).item()
    l0 = nll(W, h, x, fit_T(W, h, x)).item()
    return alpha.item(), (l0 - l1) * len(x)


def per_resp_ll(W, h, x, rid, logT, A=None, Z=None):
    with torch.no_grad():
        out = {}
        for r in rid.unique().tolist():
            m = rid == r
            out[r] = -nll(W, h[m], x[m], logT, A, None if Z is None else Z[m]).item() * int(m.sum())
        return out


def basis(data, r):
    """Label-free PCA basis of pooled hidden states -> standardized projection function."""
    H = torch.cat([v["h"] for v in data.values()])
    mu = H.mean(0)
    idx = torch.randperm(len(H))[:20000]
    U, S, V = torch.pca_lowrank(H[idx] - mu, q=r, center=False)
    P = V[:, :r]
    sd = ((H[idx] - mu) @ P).std(0)
    # constant column first: lets the model learn a global (context-independent) shift of h -> unigram-like logit bias
    return lambda h: torch.cat([torch.ones(len(h), 1, device=h.device), ((h - mu) @ P) / sd], 1)


def load(tag):
    blob = torch.load(common.DATA / f"hid_{tag}.pt")
    W = blob["W"].float().to(DEV)
    data = {ep: {"h": v["h"].float().to(DEV), "x": v["x"].to(DEV), "rid": v["rid"].to(DEV)} for ep, v in blob["data"].items()}
    return W, data


def cv(tag, r, folds=5, lr=3e-3):
    W, data = load(tag)
    proj = basis(data, r)
    print(f"LDT cross-validated gain (basis r={r}, lr={lr}); gain/tok in millinats, z over held-out responses")
    res = {}
    for ep in sorted(data):
        h, x, rid = data[ep]["h"], data[ep]["x"], data[ep]["rid"]
        Z = proj(h)
        nres = int(rid.max()) + 1
        fold_of = torch.arange(nres, device=DEV) % folds
        gains, steps = [], []
        for f in range(folds):
            tr = fold_of[rid] != f
            te = ~tr
            logT0 = fit_T(W, h[tr], x[tr])
            p1 = fit_dir(W, h[tr], x[tr], Z[tr], rid[tr], lr=lr)
            steps.append(p1["steps"])
            l0 = per_resp_ll(W, h[te], x[te], rid[te], logT0)
            l1 = per_resp_ll(W, h[te], x[te], rid[te], p1["logT"], p1["A"], Z[te])
            gains += [l1[k] - l0[k] for k in l0]
        g = np.array(gains)
        n = len(x)
        z = g.sum() / (np.sqrt((g ** 2).sum()) + 1e-9)
        res[ep] = (g.sum() / n * 1000, z)
        print(f"  {ep}: gain={g.sum() / n * 1000:+7.2f} mnat/tok  total={g.sum():+8.1f}  z={z:+.2f}  n={n}  steps={steps}", flush=True)
    return res


def transfer(tag, r, lr=3e-3):
    W, data = load(tag)
    proj = basis(data, r)
    eps = sorted(data)
    Zs = {e: proj(data[e]["h"]) for e in eps}
    dirs = {}
    for e in eps:
        p = fit_dir(W, data[e]["h"], data[e]["x"], Zs[e], data[e]["rid"], lr=lr)
        dirs[e] = p["A"]
    print("alpha[F | direction learned on E]   (rows E, cols F);  then LL gain (nats) of F along E's direction")
    M = np.zeros((len(eps), len(eps))); G = np.zeros_like(M)
    for i, e in enumerate(eps):
        for j, f in enumerate(eps):
            M[i, j], G[i, j] = fit_alpha(W, data[f]["h"], data[f]["x"], Zs[f], dirs[e])
    hdr = "        " + " ".join(f"{f[-2:]:>6}" for f in eps)
    print(hdr)
    for i, e in enumerate(eps):
        print(f"{e:>7} " + " ".join(f"{v:6.2f}" for v in M[i]))
    print(hdr)
    for i, e in enumerate(eps):
        print(f"{e:>7} " + " ".join(f"{v:6.1f}" for v in G[i]))
    np.save(common.DATA / f"transfer_{tag}_r{r}.npy", np.stack([M, G]))
    return M, G


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--fam")
    ap.add_argument("--model")
    ap.add_argument("--tag")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    a = ap.parse_args()
    if a.cmd == "extract":
        if a.fam:
            from fams import FAMILIES
            mid, keep = FAMILIES[a.fam]; tag = a.fam
        else:
            mid, keep, tag = a.model, None, a.tag
        rows = [r for d in a.rest for r in common.load(d) if keep is None or r["endpoint"] in keep]
        extract(mid, rows, tag)
    elif a.cmd == "cv":
        cv(a.rest[0], a.rank, lr=a.lr)
    elif a.cmd == "transfer":
        transfer(a.rest[0], a.rank, lr=a.lr)
