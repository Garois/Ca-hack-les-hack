"""Build per-image soft targets q(x) that imitate a model that never saw x.

q_y = p*(x) = sigmoid(a - b*u(x)), u = normal-quantile of the difficulty rank; a solved so mean CE == --mu
(over the non-removed images). Non-true mass from held-out-like confusion (--rest), ^(1/tau).
--class_mode capped: classes with count >= --cap are treated as fully removed: p* = --p_removed.
Also writes delta(x) = true-class logit shift used as teacher on augmented views: t(view) = orig(view) - delta*e_y.
--mode smooth: q = softmax(z/tau) (mean-only reference baseline).
"""
import argparse
import json

import numpy as np
import torch

from common import ART

ap = argparse.ArgumentParser()
ap.add_argument("--mu", type=float, default=0.8, help="target mean CE on (non-removed) forget images")
ap.add_argument("--b", type=float, default=1.0, help="spread of target log-odds")
ap.add_argument("--tau", type=float, default=1.0, help="soften non-true distribution")
ap.add_argument("--proxy", default="cv", choices=["aug", "cv", "comb", "none"])
ap.add_argument("--rest", default="mix", choices=["orig", "cv", "strong", "mix"])
ap.add_argument("--class_mode", default="sample", choices=["sample", "capped"])
ap.add_argument("--cap", type=int, default=129)
ap.add_argument("--p_removed", type=float, default=1e-3)
ap.add_argument("--mode", default="shift", choices=["shift", "smooth"])
ap.add_argument("--floor", type=float, default=1e-4)
ap.add_argument("--tag", required=True)
args = ap.parse_args()

S = np.load(ART / "orig_stats.npz")
y, z = S["y"], S["z_clean"].astype(np.float64)
n = len(y)
ar = np.arange(n)


def softmax(a, axis=-1):
    a = a - a.max(axis, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis, keepdims=True)


def nontrue(p):
    p = p.astype(np.float64).copy()
    p[ar, y] = 0
    return p / np.maximum(p.sum(1, keepdims=True), 1e-30)


p_clean = softmax(z)
p_base = 0.5 * p_clean + 0.5 * softmax(S["z_mild"].astype(np.float64)).mean(0)
py_base = np.clip(p_base[ar, y], 1e-6, 1 - 1e-9)
counts = S["counts"]
removed = (counts[y] >= args.cap) if args.class_mode == "capped" else np.zeros(n, bool)

if args.mode == "smooth":
    q = softmax(z / args.tau)
    pstar = q[ar, y]
else:
    rests = {"orig": lambda: nontrue(p_base), "cv": lambda: nontrue(S["P_cv"]),
             "strong": lambda: nontrue(S["p_strong"]),
             "mix": lambda: 0.5 * nontrue(S["P_cv"]) + 0.5 * nontrue(S["p_strong"])}
    rest = rests[args.rest]()
    if args.tau != 1.0:
        rest = np.power(np.maximum(rest, 1e-12), 1.0 / args.tau)
        rest[ar, y] = 0
        rest /= rest.sum(1, keepdims=True)

    if args.proxy == "aug":
        s = S["s_aug"]
    elif args.proxy == "cv":
        s = S["s_cv"]
    elif args.proxy == "comb":
        s = (np.argsort(np.argsort(S["s_aug"])) + np.argsort(np.argsort(S["s_cv"]))) / 2.0
    else:
        s = np.zeros(n)
    rank = np.argsort(np.argsort(s + 1e-9 * ar))
    u = torch.special.ndtri(torch.as_tensor((rank + 0.5) / n, dtype=torch.float64)).numpy()
    if args.proxy == "none":
        u = np.zeros(n)

    def build(a):
        ps = 1 / (1 + np.exp(-(a - args.b * u)))
        ps = np.minimum(ps, py_base)  # never raise confidence
        return np.where(removed, args.p_removed, ps)

    keep = ~removed
    lo, hi = -30.0, 30.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if -np.log(build(mid)[keep]).mean() > args.mu:
            lo = mid
        else:
            hi = mid
    pstar = build((lo + hi) / 2)
    q = rest * (1 - pstar)[:, None]
    q[ar, y] = pstar

q = np.maximum(q, args.floor)
q /= q.sum(1, keepdims=True)
pfin = q[ar, y]
delta = np.maximum(np.log(py_base / (1 - py_base)) - np.log(pfin / (1 - pfin)), 0)
ce = -np.log(pfin)
acc = q.argmax(1) == y


def grp(m):
    return {"n": int(m.sum()), "ce_mean": float(ce[m].mean()) if m.any() else None,
            "ce_std": float(ce[m].std()) if m.any() else None, "acc": float(acc[m].mean()) if m.any() else None}


info = {
    "tag": args.tag, **vars(args),
    "target_ce_mean": float(ce.mean()), "target_ce_std": float(ce.std()), "target_acc": float(acc.mean()),
    "kept": grp(~removed), "removed": grp(removed),
    "orig_ce_mean": float(-np.log(p_clean[ar, y]).mean()),
    "delta_mean": float(delta.mean()), "frac_delta_pos": float((delta > 1e-6).mean()),
    "finite": bool(np.isfinite(q).all()), "min_q": float(q.min()),
    "wrong_pick_eq_probe_top_other": float((q.argmax(1)[~acc] == nontrue(S["P_cv"]).argmax(1)[~acc]).mean())
    if (~acc).any() else None,
}
assert info["finite"]
np.savez_compressed(ART / f"targets_{args.tag}.npz", q=q.astype(np.float32), y=y, delta=delta.astype(np.float32),
                    pstar=pfin.astype(np.float32), removed=removed)
(ART / f"targets_{args.tag}.json").write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2))
