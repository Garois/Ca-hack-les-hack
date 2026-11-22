"""Fast learned-direction test with a global direction only (rank 0 = constant shift a of the final hidden state,
i.e. a logit bias w = W a over the vocabulary), plus temperature.

    logits_t = (W h_t + alpha * W a) / T

cv      : per endpoint, K-fold held-out gain of learning a (vs temperature only)  -> "does it deviate from base?"
transfer: learn a_E on each endpoint E (all data); for every endpoint F fit (T, alpha) along w_E; report alpha and the
          LL gain (nats) over temperature-only. Under H0 (F = base along that direction) gain ~ chi2_1 / 2.

Usage: python ldt0.py TAG [--folds 5] [--lr 3e-3]       (TAG = hid_<TAG>.pt made by ldt.py extract)
"""
import argparse
import math

import numpy as np
import torch

import common

DEV = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True


def load(tag):
    blob = torch.load(common.DATA / f"hid_{tag}.pt")
    W = blob["W"].float().to(DEV)
    data = {}
    for ep, v in blob["data"].items():
        h = v["h"].float().to(DEV)
        data[ep] = {"L0": (h @ W.T).to(torch.bfloat16), "x": v["x"].to(DEV), "rid": v["rid"].to(DEV)}
        del h
    return W, data


def nll(L0, x, logT, w=None, alpha=None, bs=2048):
    tot = 0.0
    for i in range(0, len(x), bs):
        lg = L0[i:i + bs].float()
        if w is not None:
            lg = lg + (w if alpha is None else alpha * w)
        tot = tot + torch.nn.functional.cross_entropy(lg * torch.exp(-logT), x[i:i + bs], reduction="sum")
    return tot / len(x)


def fit_T(L0, x, steps=50):
    logT = torch.tensor(math.log(0.7), device=DEV, requires_grad=True)
    opt = torch.optim.Adam([logT], lr=0.05)
    for _ in range(steps):
        opt.zero_grad(); loss = nll(L0, x, logT); loss.backward(); opt.step()
    return logT.detach()


def fit_dir(W, L0, x, rid, lr=3e-3, max_steps=200, every=10, val_frac=0.2, seed=0):
    g = torch.Generator().manual_seed(seed)
    ur = rid.unique().cpu()
    perm = ur[torch.randperm(len(ur), generator=g)].to(DEV)
    va = torch.isin(rid, perm[:max(1, int(len(ur) * val_frac))])
    tr = ~va
    logT = fit_T(L0[tr], x[tr]).clone().requires_grad_(True)
    a = torch.zeros(W.shape[1], device=DEV, requires_grad=True)
    opt = torch.optim.Adam([logT, a], lr=lr)
    with torch.no_grad():
        best = nll(L0[va], x[va], logT, W @ a).item()
    state = (logT.detach().clone(), a.detach().clone(), 0)
    for step in range(1, max_steps + 1):
        opt.zero_grad()
        loss = nll(L0[tr], x[tr], logT, W @ a)
        loss.backward()
        opt.step()
        if step % every == 0:
            with torch.no_grad():
                v = nll(L0[va], x[va], logT, W @ a).item()
            if v < best:
                best, state = v, (logT.detach().clone(), a.detach().clone(), step)
    return {"logT": state[0], "a": state[1], "steps": state[2]}


def fit_alpha(L0, x, w, steps=60):
    logT = torch.tensor(math.log(0.7), device=DEV, requires_grad=True)
    alpha = torch.zeros((), device=DEV, requires_grad=True)
    opt = torch.optim.Adam([logT, alpha], lr=0.05)
    for _ in range(steps):
        opt.zero_grad(); loss = nll(L0, x, logT, w, alpha); loss.backward(); opt.step()
    T0 = fit_T(L0, x)
    with torch.no_grad():
        l1 = nll(L0, x, logT, w, alpha).item()
        l0 = nll(L0, x, T0).item()
    return alpha.item(), (l0 - l1) * len(x), math.exp(logT.item())


def cv(W, data, folds=5, lr=3e-3):
    out = {}
    print(f"CV held-out gain of a global direction (lr={lr}); z over held-out responses")
    for ep in sorted(data):
        L0, x, rid = data[ep]["L0"], data[ep]["x"], data[ep]["rid"]
        nres = int(rid.max()) + 1
        fold_of = torch.arange(nres, device=DEV) % folds
        gains, steps = [], []
        for f in range(folds):
            te = fold_of[rid] == f
            tr = ~te
            T0 = fit_T(L0[tr], x[tr])
            p = fit_dir(W, L0[tr], x[tr], rid[tr], lr=lr)
            steps.append(p["steps"])
            w = W @ p["a"]
            with torch.no_grad():
                for r in rid[te].unique().tolist():
                    m = rid == r
                    n = int(m.sum())
                    gains.append((nll(L0[m], x[m], T0).item() - nll(L0[m], x[m], p["logT"], w).item()) * n)
        g = np.array(gains)
        z = g.sum() / (np.sqrt((g ** 2).sum()) + 1e-9)
        T = math.exp(fit_T(L0, x).item())
        out[ep] = {"gain": g.sum(), "z": z, "T": T, "n": len(x)}
        print(f"  {ep}: T={T:.3f}  gain={g.sum():+8.1f} nats ({g.sum() / len(x) * 1000:+6.2f} mnat/tok)  z={z:+.2f}  "
              f"n={len(x)}  steps={steps}", flush=True)
    return out


def transfer(W, data, lr=3e-3):
    eps = sorted(data)
    ws = {}
    for e in eps:
        p = fit_dir(W, data[e]["L0"], data[e]["x"], data[e]["rid"], lr=lr)
        ws[e] = W @ p["a"] if p["steps"] > 0 else None
    M = np.full((len(eps), len(eps)), np.nan); G = np.full_like(M, np.nan)
    for i, e in enumerate(eps):
        if ws[e] is None:
            continue
        for j, f in enumerate(eps):
            M[i, j], G[i, j], _ = fit_alpha(data[f]["L0"], data[f]["x"], ws[e])
    hdr = "         " + " ".join(f"{f[-2:]:>6}" for f in eps)
    print("alpha  (rows: direction learned on E; cols: endpoint F)\n" + hdr)
    for i, e in enumerate(eps):
        print(f"{e:>8} " + " ".join(f"{v:6.2f}" if not np.isnan(v) else "     ." for v in M[i]))
    print("gain (nats)\n" + hdr)
    for i, e in enumerate(eps):
        print(f"{e:>8} " + " ".join(f"{v:6.1f}" if not np.isnan(v) else "     ." for v in G[i]), flush=True)
    return eps, M, G


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--no-cv", action="store_true")
    a = ap.parse_args()
    W, data = load(a.tag)
    res = {"cv": None}
    if not a.no_cv:
        res["cv"] = cv(W, data, a.folds, a.lr)
    eps, M, G = transfer(W, data, a.lr)
    res.update({"eps": eps, "alpha": M, "gain": G})
    import pickle
    with open(common.DATA / f"ldt0_{a.tag}.pkl", "wb") as f:
        pickle.dump(res, f)
