"""Targets v2 for the *kept* forget images (classes not treated as fully removed).

p*(x): logit(p*) = a + b * log(n_keep(c) / median n_keep), n_keep(c) = --class_size - count(c)  (fewer images left -> less
confident), a solved so mean CE over kept images == --mu.  (--b 0 -> same confidence for every class.)
Non-true mass over all 100 classes: orig's softmax on clean+mild views and strong views, true class masked, ^(1/tau).
Removed images (count >= --cap) get q = orig clean softmax (they are handled by head-bias surgery, see head_rm.py).
"""
import argparse
import json

import numpy as np

from common import ART

ap = argparse.ArgumentParser()
ap.add_argument("--mu", type=float, default=0.7)
ap.add_argument("--b", type=float, default=1.0)
ap.add_argument("--tau", type=float, default=2.0)
ap.add_argument("--w_strong", type=float, default=0.5)
ap.add_argument("--cap", type=int, default=129, help="count >= cap -> removed class; use 10**9 for none")
ap.add_argument("--class_size", type=int, default=130)
ap.add_argument("--floor", type=float, default=1e-5)
ap.add_argument("--b_img", type=float, default=0.0, help="per-image spread from difficulty proxy")
ap.add_argument("--img_proxy", default="cv", choices=["cv", "aug"])
ap.add_argument("--mode", default="sigmoid", choices=["sigmoid", "probe"])
ap.add_argument("--eps", type=float, default=0.1, help="probe mode: mass from orig non-true distribution")
ap.add_argument("--t", type=float, default=1.0, help="probe mode: temperature on probe probabilities")
ap.add_argument("--tag", required=True)
args = ap.parse_args()

S = np.load(ART / "orig_stats.npz")
y, counts = S["y"], S["counts"]
n = len(y)
ar = np.arange(n)


def softmax(a):
    a = a - a.max(-1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(-1, keepdims=True)


def nontrue(p):
    p = p.astype(np.float64).copy()
    p[ar, y] = 0
    return p / np.maximum(p.sum(1, keepdims=True), 1e-30)


p_clean = softmax(S["z_clean"].astype(np.float64))
p_mild = softmax(S["z_mild"].astype(np.float64)).mean(0)
rest = (1 - args.w_strong) * nontrue(0.5 * p_clean + 0.5 * p_mild) + args.w_strong * nontrue(S["p_strong"])
rest = np.power(np.maximum(rest, 1e-30), 1.0 / args.tau)
rest[ar, y] = 0
rest /= rest.sum(1, keepdims=True)

removed = counts[y] >= args.cap
kept = ~removed
n_keep = np.maximum(args.class_size - counts[y], 5).astype(np.float64)
v = np.log(n_keep / np.median(n_keep[kept]))
py_orig = np.clip(0.5 * p_clean[ar, y] + 0.5 * p_mild[ar, y], 1e-6, 1 - 1e-9)


if args.b_img > 0:
    import torch
    s = S["s_cv"] if args.img_proxy == "cv" else S["s_aug"]
    rank = np.argsort(np.argsort(s + 1e-9 * ar))
    u = torch.special.ndtri(torch.as_tensor((rank + 0.5) / n, dtype=torch.float64)).numpy()
else:
    u = np.zeros(n)


def build(a):
    return np.minimum(1 / (1 + np.exp(-(a + args.b * v - args.b_img * u))), py_orig)


if args.mode == "probe":
    # distill toward the out-of-fold linear probe (a predictor that never saw x), sharpened/softened by t so that
    # mean CE == mu; eps of mass from orig's 100-class non-true distribution
    # mean CE is set monotonically by blending with orig (w -> 1 gives orig's near-zero CE); --t softens the probe
    P = np.power(np.maximum(S["P_cv"].astype(np.float64), 1e-12), 1.0 / args.t)
    P /= P.sum(1, keepdims=True)
    p_o = 0.5 * p_clean + 0.5 * p_mild

    def build_q(w):
        return (1 - args.eps) * ((1 - w) * P + w * p_o) + args.eps * rest

    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if -np.log(np.maximum(build_q(mid)[ar, y], args.floor))[kept].mean() > args.mu:
            lo = mid
        else:
            hi = mid
    q = build_q((lo + hi) / 2)
    print("probe blend w_orig =", float((lo + hi) / 2))
else:
    lo, hi = -30.0, 30.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if -np.log(build(mid)[kept]).mean() > args.mu:
            lo = mid
        else:
            hi = mid
    ps = build((lo + hi) / 2)
    q = rest * (1 - ps)[:, None]
    q[ar, y] = ps
q[removed] = p_clean[removed]
q = np.maximum(q, args.floor)
q /= q.sum(1, keepdims=True)

ce = -np.log(q[ar, y])
acc = q.argmax(1) == y
wrong_in_forget = (counts[q.argmax(1)] > 0)[~acc & kept]
info = {"tag": args.tag, **vars(args), "n_kept": int(kept.sum()), "n_removed": int(removed.sum()),
        "kept_ce_mean": float(ce[kept].mean()), "kept_ce_std": float(ce[kept].std()),
        "kept_acc": float(acc[kept].mean()),
        "kept_wrong_pick_in_forget_classes": float(wrong_in_forget.mean()) if wrong_in_forget.size else None,
        "rest_entropy_mean": float(-(rest * np.log(np.maximum(rest, 1e-30))).sum(1)[kept].mean()),
        "finite": bool(np.isfinite(q).all())}
assert info["finite"]
np.savez_compressed(ART / f"tv2_{args.tag}.npz", q=q.astype(np.float32), y=y, removed=removed)
(ART / f"tv2_{args.tag}.json").write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2))
