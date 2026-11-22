"""PLAN_V9 B4 evaluation: nested-CV effect of 4-shift TTA, with the current decision (base) and with the calibrated
expected-mIoU decoding (emiou). Sanity check first: no-TTA probabilities must reproduce the stored OOF ones."""
from pathlib import Path

import numpy as np

from common import load_masks128
from decision import apply_iso, boot, clean_full_or_partial, emiou_mask, fit_iso, oof_probs
from map_rules import is_full
from oof import postproc
from ps_probe import subtype
from train_seg import folds
from wmloc.utils import miou_image

TTA = Path.home() / "wm_out" / "tta"; gt = load_masks128()


def load(kind, m, pat, which):
    d = {}
    for f in sorted(TTA.glob(f"{kind}_{m}_{pat}_{which}.npz")):
        z = np.load(f)
        for k in z.files:
            d[k] = z[k].astype(np.float32)
    return d


def maps(m, tta):
    suf = "|tta" if tta else ""
    if m == "stegastamp":
        U = load("unet", m, "*_f[0-4]_s2", "oof"); return {k: U[k + suf] for k in U if "|" not in k}
    pre = "" if m == "pixelseal" else "tm_"
    P = load("probe", m, f"{pre}fx[0-4]", "oof")
    if m == "pixelseal":
        return {k: P[k + suf] for k in P if "|" not in k}
    U = load("unet", m, "*_f[0-4]_s2", "oof")
    return {k: 0.75 * P[k + suf] + 0.25 * U[k + suf] for k in P if "|" not in k}


def nested_eval(m, P, items, fold_of, family):
    names = [x["image_name"] for x in items]; X = {x["image_name"]: x for x in items}
    fullp = {n: is_full(X[n]) for n in names}
    if family == "base":
        cfgs = [(t, c) for t in (0.3, 0.4, 0.5, 0.6, 0.7) for c in (0, 16)]
        sc = lambda n, c: miou_image(postproc(P[n], fullp[n], *c), gt[n])
    else:
        cal = {}
        for f in range(5):
            for full in (True, False):
                tr = [n for n in names if fold_of[n] != f and fullp[n] == full]
                cal[(f, full)] = fit_iso(np.concatenate([P[n].ravel() for n in tr]), np.concatenate([gt[n].ravel() for n in tr]).astype(float))
        Q = {n: apply_iso(cal[(fold_of[n], fullp[n])], P[n]) for n in names}
        from scipy import ndimage
        cfgs = [(s, c) for s in (0, 1) for c in (0, 16)]
        memo = {}

        def sc(n, c):
            if (n, c) not in memo:
                q = ndimage.gaussian_filter(Q[n], c[0]) if c[0] else Q[n]
                memo[(n, c)] = miou_image(clean_full_or_partial(emiou_mask(q), c[1]), gt[n])
            return memo[(n, c)]
    out = {}
    for f in range(5):
        for full in (True, False):
            tr = [n for n in names if fold_of[n] != f and fullp[n] == full]
            te = [n for n in names if fold_of[n] == f and fullp[n] == full]
            best = max(cfgs, key=lambda c: np.mean([sc(n, c) for n in tr]))
            for n in te:
                out[n] = sc(n, best)
    return out


def main():
    for m in ("stegastamp", "trustmark", "pixelseal"):
        items, fold_of = folds(m); names = [x["image_name"] for x in items]; X = {x["image_name"]: x for x in items}
        P0, P1 = maps(m, False), maps(m, True); old = oof_probs(m)
        dev = np.mean([np.abs(P0[n] - old[n]).mean() for n in names])
        print(f"== {m}: sanity |no-TTA - stored OOF| mean abs diff = {dev:.4f}")
        r = {(fam, t): nested_eval(m, P, items, fold_of, fam) for fam in ("base", "emiou") for t, P in ((0, P0), (1, P1))}
        ref = r[("base", 0)]
        for (fam, t), v in r.items():
            d = [v[n] - ref[n] for n in names]; mu, lo, hi = boot(d)
            by = {}
            for n in names:
                by.setdefault(subtype(X[n], gt[n]), []).append(v[n] - ref[n])
            print(f"   {fam:5s} TTA={t}: nested {np.mean(list(v.values())):.4f} | vs base/noTTA {mu:+.4f} (95% CI {lo:+.4f}..{hi:+.4f}) | "
                  + " ".join(f"{s}:{np.mean(x):+.3f}" for s, x in sorted(by.items())))


if __name__ == "__main__":
    main()
