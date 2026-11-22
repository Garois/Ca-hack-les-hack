"""Phase A of PLAN_V9: decision layer on top of the v8 probability maps, evaluated by NESTED cross-validation.
Families (per method; every parameter is chosen on the 4 other folds, per predicted regime when relevant):
  base   : current rule = probability > t, small-component cleanup c                 (t in T, c in C, per regime)
  emiou  : isotonic calibration (fit on the 4 other folds, per regime) -> keep the top-K cells maximising the expected
           mIoU 1/2[S_K/(K+T-S_K) + (N-K-T+S_K)/(N-S_K)] (regime-agnostic), optional pre-smoothing s, cleanup c
  area   : calibration -> keep the top round(sum q) cells (expected area), smoothing s, cleanup c
  +rect  : on top of the best of the above, partial-looking masks (fraction < 0.5) whose main region fills >= tau of
           its bounding box are replaced by the box of the q >= q_low region containing it
Prints nested-CV means, paired differences vs base with bootstrap 95% CI, per subtype."""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from common import load_masks128
from map_rules import is_full
from oof import postproc, valprobs
from ps_probe import subtype
from train_seg import folds
from wmloc.utils import miou_image

PROBE = Path.home() / "wm_out" / "ps_probe"; gt = load_masks128(); N = 128 * 128
T_GRID = (0.3, 0.4, 0.5, 0.6, 0.7); C_GRID = (0, 16); S_GRID = (0, 1)


def lp(p):
    d = {}
    for f in sorted(PROBE.glob(p)):
        z = np.load(f); d.update({k: z[k].astype(np.float32) for k in z.files})
    return d


def oof_probs(m):
    U = {}
    for f in range(5):
        U.update(valprobs(f"{m}_f{f}_s2"))
    if m == "pixelseal":
        return lp("heldprobs_fx*.npz")
    if m == "trustmark":
        P = lp("heldprobs_tm_fx*.npz"); return {k: 0.75 * P[k] + 0.25 * U[k] for k in P}
    return U


# ---------------- calibration: monotone (pool-adjacent-violators on quantile bins), piecewise-linear
def fit_iso(p, y, nb=100):
    qs = np.unique(np.quantile(p, np.linspace(0, 1, nb + 1)))
    idx = np.clip(np.searchsorted(qs, p, side="right") - 1, 0, len(qs) - 2)
    cnt = np.bincount(idx, minlength=len(qs) - 1).astype(float); sy = np.bincount(idx, weights=y, minlength=len(qs) - 1)
    xs = np.bincount(idx, weights=p, minlength=len(qs) - 1)
    keep = cnt > 0; cnt, sy, xs = cnt[keep], sy[keep], xs[keep]
    # pool adjacent violators
    vals, wts, xm = list(sy / cnt), list(cnt), list(xs / cnt); i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1]:
            w = wts[i] + wts[i + 1]; v = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / w
            x = (xm[i] * wts[i] + xm[i + 1] * wts[i + 1]) / w
            vals[i:i + 2] = [v]; wts[i:i + 2] = [w]; xm[i:i + 2] = [x]; i = max(i - 1, 0)
        else:
            i += 1
    return np.array(xm), np.array(vals)


def apply_iso(cal, p):
    return np.interp(p, cal[0], cal[1]).astype(np.float32)


# ---------------- decoders
def emiou_mask(q):
    f = q.ravel(); order = np.argsort(-f); cs = np.concatenate([[0.0], np.cumsum(f[order])]); T = cs[-1]
    K = np.arange(N + 1, dtype=np.float64); S = cs
    u1 = K + T - S; e1 = np.where(u1 > 1e-9, S / np.maximum(u1, 1e-9), 1.0)
    u0 = N - S; e0 = np.where(u0 > 1e-9, (N - K - T + S) / np.maximum(u0, 1e-9), 1.0)
    k = int(np.argmax(e1 + e0)); m = np.zeros(N, np.uint8); m[order[:k]] = 1
    return m.reshape(128, 128)


def area_mask(q):
    f = q.ravel(); k = int(round(float(f.sum()))); order = np.argsort(-f); m = np.zeros(N, np.uint8); m[order[:k]] = 1
    return m.reshape(128, 128)


def clean_full_or_partial(m, c):
    if c <= 0:
        return m
    full = m.mean() >= 0.5
    from oof import clean
    return (1 - clean(1 - m, c)) if full else clean(m, c)


def rectify(m, q, tau, qlow):
    if m.mean() >= 0.5 or not m.any():
        return m
    lab, k = ndimage.label(m)
    big = lab == (1 + np.argmax(ndimage.sum(m, lab, range(1, k + 1))))
    ys, xs = np.where(big)
    if big.sum() / ((ys.ptp() + 1) * (xs.ptp() + 1)) < tau:
        return m
    low, _ = ndimage.label(q >= qlow); lid = np.bincount(low[big]).argmax() if (low[big] > 0).any() else 0
    reg = (low == lid) if lid > 0 else big
    ys, xs = np.where(reg); out = m.copy(); out[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 1
    return out


def boot(d, n=3000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d)
    s = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def run(m):
    items, fold_of = folds(m); P = oof_probs(m)
    names = [x["image_name"] for x in items]; X = {x["image_name"]: x for x in items}
    fullp = {n: is_full(X[n]) for n in names}
    # per held fold: calibration maps per regime fitted on the 4 other folds
    cal = {}
    for f in range(5):
        for full in (True, False):
            tr = [n for n in names if fold_of[n] != f and fullp[n] == full]
            cal[(f, full)] = fit_iso(np.concatenate([P[n].ravel() for n in tr]), np.concatenate([gt[n].ravel() for n in tr]).astype(float))
    Q = {n: apply_iso(cal[(fold_of[n], fullp[n])], P[n]) for n in names}            # out-of-fold calibrated maps

    def smooth(a, s):
        return ndimage.gaussian_filter(a, s) if s else a

    fam = {
        "base": [("base", t, c) for t in T_GRID for c in C_GRID],
        "emiou": [("emiou", s, c) for s in S_GRID for c in C_GRID],
        "area": [("area", s, c) for s in S_GRID for c in C_GRID],
    }
    cache = {}

    def mask(n, cfg):
        key = (n, cfg)
        if key not in cache:
            if cfg[0] == "base":
                cache[key] = postproc(P[n], fullp[n], cfg[1], cfg[2])
            elif cfg[0] == "emiou":
                cache[key] = clean_full_or_partial(emiou_mask(smooth(Q[n], cfg[1])), cfg[2])
            elif cfg[0] == "area":
                cache[key] = clean_full_or_partial(area_mask(smooth(Q[n], cfg[1])), cfg[2])
        return cache[key]

    def score(n, cfg):
        if cfg[0] == "rect":
            inner, tau, ql = cfg[1], cfg[2], cfg[3]
            return miou_image(rectify(mask(n, inner), Q[n], tau, ql), gt[n])
        return miou_image(mask(n, cfg), gt[n])

    def nested(cfgs, per_regime=True):
        out = {}
        for f in range(5):
            for full in ((True, False) if per_regime else (None,)):
                tr = [n for n in names if fold_of[n] != f and (full is None or fullp[n] == full)]
                te = [n for n in names if fold_of[n] == f and (full is None or fullp[n] == full)]
                best = max(cfgs, key=lambda c: np.mean([score(n, c) for n in tr]))
                for n in te:
                    out[n] = score(n, best)
        return out

    res = {k: nested(v) for k, v in fam.items()}
    best_fam = max(("emiou", "area"), key=lambda k: np.mean(list(res[k].values())))
    rect_cfgs = [("rect", cf, tau, ql) for cf in fam[best_fam] for tau in (0.7, 0.8, 0.9) for ql in (0.15, 0.3)] + fam[best_fam]
    res["rect+" + best_fam] = nested(rect_cfgs)
    print(f"== {m} (n={len(names)})")
    for k, r in res.items():
        d = [r[n] - res["base"][n] for n in names]
        mu, lo, hi = boot(d)
        by = {}
        for n in names:
            by.setdefault(subtype(X[n], gt[n]), []).append(r[n] - res["base"][n])
        print(f"   {k:12s} nested mIoU {np.mean([r[n] for n in names]):.4f} | vs base {mu:+.4f} (95% CI {lo:+.4f}..{hi:+.4f}) | "
              + " ".join(f"{s}:{np.mean(v):+.3f}" for s, v in sorted(by.items())))
    # regime errors: images whose predicted regime is wrong
    wrong = [n for n in names if fullp[n] != bool(X[n]["full_watermark"])]
    if wrong:
        print(f"   regime-misclassified images (n={len(wrong)}): base {np.mean([res['base'][n] for n in wrong]):.3f} -> "
              f"{best_fam} {np.mean([res[best_fam][n] for n in wrong]):.3f}")
    return res


if __name__ == "__main__":
    for m in ("stegastamp", "trustmark", "pixelseal"):
        run(m)
