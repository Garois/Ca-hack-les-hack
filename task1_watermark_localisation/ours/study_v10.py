"""Study after v9 (CPU, OOF TTA maps): (1) error budget per method x subtype with the v9 decision, (2) mIoU vs region
size (are small holes/objects hopeless?) with per-image oracle threshold, (3) StegaStamp rectangle seed quality with
v9 maps (centre error within 12 / 24 px, size error)."""
import numpy as np
from scipy import ndimage

from common import load_mask512, load_masks128
from decision import apply_iso, emiou_mask, fit_iso, clean_full_or_partial
from map_rules import is_full
from ps_probe import subtype
from train_seg import folds
from tta_eval import maps as oof_maps
from wmloc.utils import miou_image

gt = load_masks128()
CFG = {"pixelseal": {True: (1, 16), False: (1, 16)}, "trustmark": {True: (1, 16), False: (1, 16)},
       "stegastamp": {True: (1, 16), False: (0, 0)}}


def decode(q, s, c):
    return clean_full_or_partial(emiou_mask(ndimage.gaussian_filter(q, s) if s else q), c)


for m in ("pixelseal", "trustmark", "stegastamp"):
    items, fold_of = folds(m); P = oof_maps(m, True); names = [x["image_name"] for x in items]; X = {x["image_name"]: x for x in items}
    fullp = {n: is_full(X[n]) for n in names}; Q = {}
    for f in range(5):
        for full in (True, False):
            tr = [n for n in names if fold_of[n] != f and fullp[n] == full]
            cal = fit_iso(np.concatenate([P[n].ravel() for n in tr]), np.concatenate([gt[n].ravel() for n in tr]).astype(float))
            for n in names:
                if fold_of[n] == f and fullp[n] == full:
                    Q[n] = apply_iso(cal, P[n])
    rows = []
    for n in names:
        g = gt[n]; mk = decode(Q[n], *CFG[m][fullp[n]]); st = subtype(X[n], g)
        small = (1 - g) if X[n]["full_watermark"] else g
        orc = max(miou_image((Q[n] >= t).astype(np.uint8), g) for t in np.unique(np.quantile(Q[n], np.linspace(0, 1, 101))))
        rows.append((st, small.mean(), miou_image(mk, g), orc))
    print(f"== {m}")
    for st in sorted({r[0] for r in rows}):
        v = [r for r in rows if r[0] == st]
        print(f"   {st:15s} n={len(v):3d} v9 {np.mean([r[2] for r in v]):.3f} | oracle-thr {np.mean([r[3] for r in v]):.3f} | loss share {sum(1 - r[2] for r in v) / len(rows):.3f}")
    for st in ("full/object", "partial/object"):
        v = np.array([(r[1], r[2], r[3]) for r in rows if r[0] == st])
        if not len(v):
            continue
        edges = [0, 0.02, 0.05, 0.1, 0.2, 1.0]
        parts = []
        for a, b in zip(edges[:-1], edges[1:]):
            s = v[(v[:, 0] >= a) & (v[:, 0] < b)]
            if len(s):
                parts.append(f"[{a:.2f},{b:.2f}) n{len(s)} v9 {s[:, 1].mean():.2f}/orc {s[:, 2].mean():.2f}")
        print(f"   {st} by region size: " + " | ".join(parts))
    if m == "stegastamp":
        res = []
        for n in names:
            if subtype(X[n], gt[n]) != "partial/rect":
                continue
            g = load_mask512(n); gy, gx = np.where(g); gb = np.array([gy.min(), gy.max() + 1, gx.min(), gx.max() + 1])
            mk = decode(Q[n], *CFG[m][fullp[n]])
            if not mk.any() or mk.mean() >= 0.5:
                res.append((np.nan, np.nan)); continue
            lab, k = ndimage.label(mk); big = lab == (1 + np.argmax(ndimage.sum(mk, lab, range(1, k + 1))))
            ys, xs = np.where(big); b = np.array([ys.min(), ys.max() + 1, xs.min(), xs.max() + 1]) * 4
            ce = max(abs((b[0] + b[1] - gb[0] - gb[1]) / 2), abs((b[2] + b[3] - gb[2] - gb[3]) / 2))
            se = max(abs((b[1] - b[0]) / (gb[1] - gb[0]) - 1), abs((b[3] - b[2]) / (gb[3] - gb[2]) - 1))
            res.append((ce, se))
        r = np.array(res, dtype=float)
        print(f"   SS rect seeds (v9 maps, all {len(r)} rects): centre err median {np.nanmedian(r[:, 0]):.0f}px, <=12px {np.nanmean(r[:, 0] <= 12):.2f}, "
              f"<=24px {np.nanmean(r[:, 0] <= 24):.2f}; relative size err median {np.nanmedian(r[:, 1]):.2f}, <=0.3 {np.nanmean(r[:, 1] <= 0.3):.2f}, "
              f"<=0.6 {np.nanmean(r[:, 1] <= 0.6):.2f}; both(<=24px & <=0.6) {np.nanmean((r[:, 0] <= 24) & (r[:, 1] <= 0.6)):.2f}")
