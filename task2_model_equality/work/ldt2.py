"""Learned-direction transfer test relative to the FAMILY CONSENSUS (removes offsets shared by all endpoints, e.g.
server-vs-local numeric/template differences). Top-K compressed logits (memory friendly).

1. consensus: fit one global logit-bias c = W a_c (+ per-endpoint T) on ALL endpoints of the family.
   New baseline: logits0 = L0 + c.
2. cv: per endpoint, K-fold held-out gain of an extra global direction (deviation from consensus).
3. transfer: direction a_E learned on each E (relative to consensus, early-stopped); for every F, fit (T, alpha)
   along w_E: alpha matrix and LL gain (nats). H0 gain ~ chi2_1/2.

Usage: python ldt2.py TAG [--topk 1024] [--lr 3e-3] [--no-cv]
"""
import argparse
import math
import pickle

import numpy as np
import torch

import common
from ldt1 import load

DEV = "cuda"


def nll_sum(d, m, logT, base=None, wb=None, alpha=None, bs=8192):
    v, ix, y = (d["v"], d["i"], d["y"]) if m is None else (d["v"][m], d["i"][m], d["y"][m])
    tot = 0.0
    for s in range(0, len(y), bs):
        lg = v[s:s + bs].float()
        if base is not None or wb is not None:
            ii = ix[s:s + bs].long()
            if base is not None:
                lg = lg + base[ii]
            if wb is not None:
                lg = lg + (wb[ii] if alpha is None else alpha * wb[ii])
        tot = tot + torch.nn.functional.cross_entropy(lg * torch.exp(-logT), y[s:s + bs].long(), reduction="sum")
    return tot


def fit_T(d, m=None, base=None, steps=40):
    logT = torch.tensor(math.log(0.7), device=DEV, requires_grad=True)
    opt = torch.optim.Adam([logT], lr=0.05)
    n = int(m.sum()) if m is not None else d["n"]
    for _ in range(steps):
        opt.zero_grad(); (nll_sum(d, m, logT, base) / n).backward(); opt.step()
    return logT.detach()


def split(d, seed=0, frac=0.2):
    g = torch.Generator().manual_seed(seed)
    ur = d["rid"].unique().cpu()
    perm = ur[torch.randperm(len(ur), generator=g)].to(DEV)
    va = torch.isin(d["rid"], perm[:max(1, int(len(ur) * frac))])
    return ~va, va


def fit_bias(W, ds, masks, base=None, lr=3e-3, max_steps=300, every=10):
    """Fit one global direction a (shared, alpha fixed = 1) + per-dataset T on datasets ds (list), with masks
    [(train, val)] per dataset; early stopping on the val parts. Returns w = W a (V,) and steps."""
    a = torch.zeros(W.shape[1], device=DEV, requires_grad=True)
    logT = [fit_T(d, tr, base).clone().requires_grad_(True) for d, (tr, va) in zip(ds, masks)]
    opt = torch.optim.Adam([a] + logT, lr=lr)
    ntr = sum(int(tr.sum()) for tr, _ in masks)

    def vloss():
        with torch.no_grad():
            w = W @ a
            return sum(nll_sum(d, va, lt, base, w).item() for d, (tr, va), lt in zip(ds, masks, logT))

    best, state = vloss(), (torch.zeros_like(a), 0)
    for step in range(1, max_steps + 1):
        opt.zero_grad()
        w = W @ a
        loss = sum(nll_sum(d, tr, lt, base, w) for d, (tr, va), lt in zip(ds, masks, logT)) / ntr
        loss.backward()
        opt.step()
        if step % every == 0:
            v = vloss()
            if v < best:
                best, state = v, (a.detach().clone(), step)
    return W @ state[0], state[1]


def fit_alpha(d, base, wb, m=None, steps=60):
    logT = fit_T(d, m, base).clone().requires_grad_(True)
    alpha = torch.zeros((), device=DEV, requires_grad=True)
    opt = torch.optim.Adam([logT, alpha], lr=0.05)
    n = int(m.sum()) if m is not None else d["n"]
    for _ in range(steps):
        opt.zero_grad(); (nll_sum(d, m, logT, base, wb, alpha) / n).backward(); opt.step()
    T0 = fit_T(d, m, base)
    with torch.no_grad():
        gain = nll_sum(d, m, T0, base).item() - nll_sum(d, m, logT, base, wb, alpha).item()
    return alpha.item(), gain, math.exp(logT.item())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--topk", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-cv", action="store_true")
    a = ap.parse_args()
    W, data = load(a.tag, a.topk)
    eps = sorted(data)
    out = {"eps": eps}

    # 1. consensus bias
    ds = [data[e] for e in eps]
    base, st = fit_bias(W, ds, [split(d) for d in ds], lr=a.lr)
    print(f"== {a.tag}: consensus bias fitted (steps={st}, |c|={base.norm().item():.2f})")
    for e in eps:
        T0 = fit_T(data[e]); T1 = fit_T(data[e], None, base)
        with torch.no_grad():
            g = nll_sum(data[e], None, T0).item() - nll_sum(data[e], None, T1, base).item()
        print(f"  {e}: gain from consensus bias = {g:+7.1f} nats   T={math.exp(T1.item()):.3f}")

    # 2. CV deviation from consensus
    if not a.no_cv:
        print("CV held-out gain of an own direction on top of the consensus")
        out["cv"] = {}
        for e in eps:
            d = data[e]
            nres = int(d["rid"].max()) + 1
            fold_of = torch.arange(nres, device=DEV) % a.folds
            gains, steps = [], []
            for f in range(a.folds):
                te = fold_of[d["rid"]] == f
                tr = ~te
                g = torch.Generator().manual_seed(f)
                ur = d["rid"][tr].unique().cpu()
                perm = ur[torch.randperm(len(ur), generator=g)].to(DEV)
                va_in = tr & torch.isin(d["rid"], perm[:max(1, len(ur) // 5)])
                tr_in = tr & ~va_in
                w, s = fit_bias(W, [d], [(tr_in, va_in)], base=base, lr=a.lr)
                steps.append(s)
                T0 = fit_T(d, tr, base)
                # T for the alternative: refit on train with w
                lt = fit_T(d, tr, base + w)
                with torch.no_grad():
                    for r in d["rid"][te].unique().tolist():
                        m = d["rid"] == r
                        gains.append(nll_sum(d, m, T0, base).item() - nll_sum(d, m, lt, base + w).item())
            gsum = np.array(gains)
            z = gsum.sum() / (np.sqrt((gsum ** 2).sum()) + 1e-9)
            out["cv"][e] = (gsum.sum(), z)
            print(f"  {e}: CV gain={gsum.sum():+7.1f} nats  z={z:+.2f}  steps={steps}", flush=True)

    # 3. transfer
    ws = {}
    for e in eps:
        w, s = fit_bias(W, [data[e]], [split(data[e])], base=base, lr=a.lr)
        ws[e] = w if s > 0 else None
    M = np.full((len(eps), len(eps)), np.nan); G = np.full_like(M, np.nan)
    for i, e in enumerate(eps):
        if ws[e] is None:
            continue
        for j, f in enumerate(eps):
            M[i, j], G[i, j], _ = fit_alpha(data[f], base, ws[e])
    hdr = "         " + " ".join(f"{f[-2:]:>6}" for f in eps)
    print("alpha (rows: direction learned on E relative to consensus; cols: endpoint F)\n" + hdr)
    for i, e in enumerate(eps):
        print(f"{e:>8} " + " ".join(f"{v:6.2f}" if not np.isnan(v) else "     ." for v in M[i]))
    print("gain (nats)\n" + hdr)
    for i, e in enumerate(eps):
        print(f"{e:>8} " + " ".join(f"{v:6.1f}" if not np.isnan(v) else "     ." for v in G[i]), flush=True)
    out.update({"alpha": M, "gain": G})
    with open(common.DATA / f"ldt2_{a.tag}.pkl", "wb") as f:
        pickle.dump(out, f)
