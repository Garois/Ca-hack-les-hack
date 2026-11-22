"""Fit the decoding strategy (temperature T, top-p) of each endpoint under a local base model, by maximum likelihood
on its responses, and report how well the best decoding explains the tokens.

For each grid cell (T, top_p): loglik(x) = log[(1-eps) * p_{T,top_p}(x) + eps * p_T(x)]  (eps avoids -inf).
Outputs per endpoint: best T/top_p, loglik per token at best, fraction of tokens outside the top_p nucleus,
and saves the full grid to data/fit_<data>__<tag>.pkl

Usage: python decode_fit.py MODEL_ID DATA_NAME [DATA_NAME ...] [--min-tok K]
"""
import pickle
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import common
from score import build

TS = np.round(np.arange(0.2, 1.51, 0.05), 2)
PS = [0.7, 0.8, 0.9, 0.95, 1.0]
EPS = 1e-3


@torch.no_grad()
def grid_for_row(model, tok, r):
    ids_ctx = tok(build(tok, r["prompt"], None), add_special_tokens=False)["input_ids"]
    ids_resp = tok(r["response"], add_special_tokens=False)["input_ids"]
    if not ids_resp:
        return None
    ids = torch.tensor([ids_ctx + ids_resp], device="cuda")
    logits = model(ids).logits[0, len(ids_ctx) - 1:-1].float()           # [n, V]
    tgt = ids[0, len(ids_ctx):]
    n = len(tgt)
    ll = np.zeros((len(TS), len(PS)))
    outside = np.zeros((len(TS), len(PS)))
    for i, T in enumerate(TS):
        lp = torch.log_softmax(logits / float(T), -1)
        p = lp.exp()
        sp, idx = p.sort(-1, descending=True)
        csum = sp.cumsum(-1)
        # rank of target in sorted order
        rank = (p > p.gather(1, tgt[:, None])).sum(1)                          # 0-based
        cum_before = torch.where(rank > 0, csum.gather(1, (rank - 1).clamp(min=0)[:, None])[:, 0], torch.zeros_like(rank, dtype=csum.dtype))
        pt = p.gather(1, tgt[:, None])[:, 0]
        for j, P in enumerate(PS):
            if P >= 1.0:
                inside = torch.ones_like(pt, dtype=torch.bool)
                mass = torch.ones_like(pt)
            else:
                inside = cum_before < P                                          # token is in the nucleus
                # nucleus mass: smallest prefix with csum >= P
                k = (csum < P).sum(1)                                            # index of last included
                mass = csum.gather(1, k[:, None])[:, 0]
            pn = torch.where(inside, pt / mass, torch.zeros_like(pt))
            ll[i, j] = torch.log((1 - EPS) * pn + EPS * pt).sum().item()
            outside[i, j] = (~inside).sum().item()
    return ll, outside, n


if __name__ == "__main__":
    args = sys.argv[1:]
    fam = None
    if "--fam" in args:
        i = args.index("--fam"); fam = args[i + 1]; del args[i:i + 2]
    if fam:
        from fams import FAMILIES
        mid, keep = FAMILIES[fam]
        datas = args
    else:
        mid, datas, keep = args[0], args[1:], None
    rows = [r for d in datas for r in common.load(d) if keep is None or r["endpoint"] in keep]
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32, device_map="cuda").eval()
    acc = {}
    for r in rows:
        g = grid_for_row(model, tok, r)
        if g is None:
            continue
        a = acc.setdefault(r["endpoint"], [np.zeros((len(TS), len(PS))), np.zeros((len(TS), len(PS))), 0, []])
        a[0] += g[0]; a[1] += g[1]; a[2] += g[2]; a[3].append((r["id"], g[0], g[1], g[2]))
    tag = mid.split("/")[-1]
    with open(common.DATA / f"fit_{'+'.join(datas)}__{fam or tag}.pkl", "wb") as f:
        pickle.dump({"TS": TS, "PS": PS, "acc": acc}, f)
    print(f"{tag} on {datas}")
    for ep in sorted(acc):
        ll, out, n, _ = acc[ep]
        i, j = np.unravel_index(ll.argmax(), ll.shape)
        jT1 = PS.index(1.0)
        iT = ll[:, jT1].argmax()
        print(f"  {ep}: best T={TS[i]:.1f} top_p={PS[j]:.2f} ll/tok={ll[i, j] / n:+.4f} outside={out[i, j] / n:.4f}  | "
              f"best T(no top-p)={TS[iT]:.1f} ll/tok={ll[iT, jT1] / n:+.4f}  | out@p.9,T={TS[i]:.1f}: {out[i, PS.index(0.9)] / n:.4f}  n={n}")
