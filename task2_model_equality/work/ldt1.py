"""Family-level learned-direction test with leave-one-out evaluation (top-K compressed logits, memory friendly).

Model for endpoint i of a family:   logits_t = (L0_t + sum_k alpha_ik * w_k) / T_i
    L0_t = base-model logits (local), w_k = W a_k (k = 1..K_DIR global logit-bias directions shared by the family),
    alpha_i = per-endpoint strengths, T_i per-endpoint temperature.

LOO: for each endpoint j, fit {a_k, alpha_i, T_i} on all OTHER endpoints (early-stopped on held-out responses),
     then fit (alpha_j, T_j) on j alone with the directions frozen; gain_j = LL(alpha_j) - LL(alpha_j = 0) in nats.
     Under H0 (j = base model along these directions) gain_j ~ chi2_K / 2  (mean K/2, 99th pct ~ 3.3 for K=1).
Also reports the in-sample alphas of the full joint fit, and bootstrap CIs of alpha_j (over responses).

Usage: python ldt1.py TAG [--kdir 1] [--topk 1024] [--boot 200]
"""
import argparse
import math
import pickle

import numpy as np
import torch

import common

DEV = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True


def compress(h, x, W, K, bs=2048):
    vals, idxs, labels = [], [], []
    for i in range(0, len(x), bs):
        lg = h[i:i + bs].float() @ W.T
        v, ix = lg.topk(K, dim=1)
        tgt = x[i:i + bs]
        tv = lg.gather(1, tgt[:, None])
        intop = (ix == tgt[:, None])
        lab = torch.where(intop.any(1), intop.float().argmax(1), torch.full_like(tgt, K))
        extra = torch.where(intop.any(1)[:, None], torch.full_like(tv, -1e4), tv)
        vals.append(torch.cat([v, extra], 1).half())
        idxs.append(torch.cat([ix, tgt[:, None]], 1).int())
        labels.append(lab)
    return torch.cat(vals), torch.cat(idxs), torch.cat(labels)


def load(tag, K):
    blob = torch.load(common.DATA / f"hid_{tag}.pt")
    W = blob["W"].float().to(DEV)
    data = {}
    for ep, v in blob["data"].items():
        h, x = v["h"].to(DEV), v["x"].to(DEV)
        vals, idxs, lab = compress(h, x, W, K)
        data[ep] = {"v": vals, "i": idxs, "y": lab, "rid": v["rid"].to(DEV), "n": len(x)}
        del h
    torch.cuda.empty_cache()
    return W, data


def nll_sum(d, m, logT, wb=None, alpha=None, bs=8192):
    """sum NLL over the tokens selected by mask m (or all if None). wb: [V, K] bias directions, alpha: [K]."""
    v, ix, y = (d["v"], d["i"], d["y"]) if m is None else (d["v"][m], d["i"][m], d["y"][m])
    tot = 0.0
    for s in range(0, len(y), bs):
        lg = v[s:s + bs].float()
        if wb is not None:
            lg = lg + (wb[ix[s:s + bs].long()] * alpha).sum(-1)
        tot = tot + torch.nn.functional.cross_entropy(lg * torch.exp(-logT), y[s:s + bs].long(), reduction="sum")
    return tot


def fit_T(d, m=None, steps=40):
    logT = torch.tensor(math.log(0.7), device=DEV, requires_grad=True)
    opt = torch.optim.Adam([logT], lr=0.05)
    n = int(m.sum()) if m is not None else d["n"]
    for _ in range(steps):
        opt.zero_grad(); (nll_sum(d, m, logT) / n).backward(); opt.step()
    return logT.detach()


def fit_joint(W, data, eps, kdir, lr=3e-3, max_steps=300, every=10, seed=0):
    """Fit shared directions a (d x kdir) + per-endpoint alpha, T on endpoints eps; early stop on held-out responses."""
    g = torch.Generator().manual_seed(seed)
    tr, va = {}, {}
    for e in eps:
        ur = data[e]["rid"].unique().cpu()
        perm = ur[torch.randperm(len(ur), generator=g)].to(DEV)
        va[e] = torch.isin(data[e]["rid"], perm[:max(1, len(ur) // 5)])
        tr[e] = ~va[e]
    d = W.shape[1]
    a = (torch.randn(d, kdir, device=DEV, generator=torch.Generator(device=DEV).manual_seed(seed)) * 1e-3).requires_grad_(True)
    alpha = {e: torch.ones(kdir, device=DEV, requires_grad=True) for e in eps}
    logT = {e: fit_T(data[e], tr[e]).clone().requires_grad_(True) for e in eps}
    params = [a] + list(alpha.values()) + list(logT.values())
    opt = torch.optim.Adam(params, lr=lr)
    ntr = sum(int(tr[e].sum()) for e in eps)

    def val_loss():
        with torch.no_grad():
            wb = W @ a
            return sum(nll_sum(data[e], va[e], logT[e], wb, alpha[e]).item() for e in eps)

    best = val_loss()
    state = (a.detach().clone(), 0)
    for step in range(1, max_steps + 1):
        opt.zero_grad()
        wb = W @ a
        loss = sum(nll_sum(data[e], tr[e], logT[e], wb, alpha[e]) for e in eps) / ntr
        loss.backward()
        opt.step()
        if step % every == 0:
            v = val_loss()
            if v < best:
                best, state = v, (a.detach().clone(), step)
    return state[0], state[1]


def fit_alpha(d, wb, kdir, m=None, steps=60):
    logT = fit_T(d, m).clone().requires_grad_(True)
    alpha = torch.zeros(kdir, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([logT, alpha], lr=0.05)
    n = int(m.sum()) if m is not None else d["n"]
    for _ in range(steps):
        opt.zero_grad(); (nll_sum(d, m, logT, wb, alpha) / n).backward(); opt.step()
    T0 = fit_T(d, m)
    with torch.no_grad():
        gain = nll_sum(d, m, T0).item() - nll_sum(d, m, logT, wb, alpha).item()
    return alpha.detach().cpu().numpy(), gain, math.exp(logT.item())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--kdir", type=int, default=1)
    ap.add_argument("--topk", type=int, default=1024)
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-3)
    a = ap.parse_args()
    W, data = load(a.tag, a.topk)
    eps = sorted(data)
    out = {"eps": eps, "loo": {}, "full": {}}
    print(f"== {a.tag}: LOO family-direction test, kdir={a.kdir}, topk={a.topk}")
    for j in eps:
        others = [e for e in eps if e != j]
        A, steps = fit_joint(W, data, others, a.kdir, lr=a.lr)
        wb = W @ A
        al, gain, T = fit_alpha(data[j], wb, a.kdir)
        # bootstrap over responses for alpha CI (directions frozen)
        rid = data[j]["rid"]
        ur = rid.unique()
        boots = []
        for b in range(a.boot):
            pick = ur[torch.randint(len(ur), (len(ur),), device=DEV)]
            cnt = torch.bincount(pick, minlength=int(rid.max()) + 1)
            # weighted bootstrap via repetition mask is awkward -> approximate with subsample of unique picks
            m = torch.isin(rid, pick.unique())
            boots.append(fit_alpha(data[j], wb, a.kdir, m=m, steps=40)[0])
        boots = np.array(boots)
        lo, hi = np.percentile(boots, 5, axis=0), np.percentile(boots, 95, axis=0)
        out["loo"][j] = {"alpha": al, "gain": gain, "T": T, "ci": (lo, hi), "steps": steps}
        print(f"  {j}: T={T:.3f}  LOO gain={gain:+7.1f} nats  alpha={np.round(al, 3)}  90%CI=[{np.round(lo, 2)}, {np.round(hi, 2)}]  "
              f"(dir fit steps={steps})", flush=True)
    # full joint fit (in-sample) for reference
    A, steps = fit_joint(W, data, eps, a.kdir, lr=a.lr)
    wb = W @ A
    print(f"  full joint fit (steps={steps}): in-sample alpha / gain")
    for e in eps:
        al, gain, T = fit_alpha(data[e], wb, a.kdir)
        out["full"][e] = {"alpha": al, "gain": gain, "T": T}
        print(f"    {e}: alpha={np.round(al, 3)} gain={gain:+.1f}")
    with open(common.DATA / f"ldt1_{a.tag}_k{a.kdir}.pkl", "wb") as f:
        pickle.dump(out, f)
