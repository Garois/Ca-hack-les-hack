"""PLAN_V10 A: edge-aware refinement of the calibrated map with a guided filter (He et al.), guide = image luminance
averaged to the 128 grid, before the expected-mIoU decoding. Nested CV per method vs the v9 decision; per regime the
filter can be off. Targets transplanted objects (visible contours), should be neutral/off for holes."""
import numpy as np
from scipy import ndimage

from common import load_image, load_masks128
from decision import apply_iso, boot, clean_full_or_partial, emiou_mask, fit_iso
from map_rules import is_full
from ps_probe import subtype
from train_seg import folds
from tta_eval import maps as oof_maps
from wmloc.utils import miou_image

gt = load_masks128()


def box(a, r):
    return ndimage.uniform_filter(a, size=2 * r + 1, mode="reflect")


def guided(I, p, r, eps):
    mI, mp = box(I, r), box(p, r); cov = box(I * p, r) - mI * mp; var = box(I * I, r) - mI * mI
    a = cov / (var + eps); b = mp - a * mI
    return np.clip(box(a, r) * I + box(b, r), 0, 1)


def guide_of(n):
    im = load_image("train", n).astype(np.float32).mean(2) / 255
    return im.reshape(128, 4, 128, 4).mean((1, 3))


def main():
    for m in ("stegastamp", "trustmark", "pixelseal"):
        items, fold_of = folds(m); P = oof_maps(m, True); names = [x["image_name"] for x in items]; X = {x["image_name"]: x for x in items}
        fullp = {n: is_full(X[n]) for n in names}; G = {n: guide_of(n) for n in names}; Q = {}
        for f in range(5):
            for full in (True, False):
                tr = [n for n in names if fold_of[n] != f and fullp[n] == full]
                cal = fit_iso(np.concatenate([P[n].ravel() for n in tr]), np.concatenate([gt[n].ravel() for n in tr]).astype(float))
                for n in names:
                    if fold_of[n] == f and fullp[n] == full:
                        Q[n] = apply_iso(cal, P[n])
        memo = {}

        def sc(n, cfg):
            if (n, cfg) not in memo:
                gf, s, c = cfg; q = Q[n]
                if gf:
                    q = guided(G[n], q, gf[0], gf[1])
                if s:
                    q = ndimage.gaussian_filter(q, s)
                memo[(n, cfg)] = miou_image(clean_full_or_partial(emiou_mask(q), c), gt[n])
            return memo[(n, cfg)]

        base_cfgs = [(None, s, c) for s in (0, 1) for c in (0, 16)]
        gf_cfgs = base_cfgs + [((r, e), s, c) for r in (2, 4, 8) for e in (1e-3, 1e-2) for s in (0, 1) for c in (0, 16)]

        def nested(cfgs):
            out = {}
            for f in range(5):
                for full in (True, False):
                    tr = [n for n in names if fold_of[n] != f and fullp[n] == full]
                    te = [n for n in names if fold_of[n] == f and fullp[n] == full]
                    best = max(cfgs, key=lambda c: np.mean([sc(n, c) for n in tr]))
                    for n in te:
                        out[n] = sc(n, best)
            return out

        rb, rg = nested(base_cfgs), nested(gf_cfgs)
        d = [rg[n] - rb[n] for n in names]; mu, lo, hi = boot(d)
        by = {}
        for n in names:
            by.setdefault(subtype(X[n], gt[n]), []).append(rg[n] - rb[n])
        print(f"== {m}: v9 decision {np.mean(list(rb.values())):.4f} -> +guided filter {np.mean(list(rg.values())):.4f} | "
              f"diff {mu:+.4f} (95% CI {lo:+.4f}..{hi:+.4f}) | " + " ".join(f"{k}:{np.mean(v):+.3f}" for k, v in sorted(by.items())))


if __name__ == "__main__":
    main()
