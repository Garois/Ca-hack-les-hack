"""Reflection 1 check: are missed holes the flat (textureless) ones? Per GT-full image: hole recall / IoU with the v9
decision vs texture (local std of luminance, 7x7) inside the hole and outside."""
import numpy as np
from scipy import ndimage
from common import load_image, load_mask512, load_masks128
from decision import apply_iso, emiou_mask, fit_iso, clean_full_or_partial
from map_rules import is_full
from train_seg import folds
from tta_eval import maps as oof_maps
gt = load_masks128()
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
        if not X[n]["full_watermark"]:
            continue
        g = gt[n]; mk = clean_full_or_partial(emiou_mask(ndimage.gaussian_filter(Q[n], 1)), 16)
        hg, hp = g == 0, mk == 0
        rec = (hg & hp).sum() / max(hg.sum(), 1)
        lum = load_image("train", n).astype(np.float32).mean(2); mu = ndimage.uniform_filter(lum, 7)
        sd = np.sqrt(np.maximum(ndimage.uniform_filter(lum * lum, 7) - mu * mu, 0))
        h512 = load_mask512(n) == 0
        rows.append((rec, sd[h512].mean() if h512.any() else np.nan, sd[~h512].mean(), hg.mean()))
    r = np.array(rows, dtype=float); t = r[:, 1]; q = np.nanpercentile(t, [33, 66])
    parts = [f"texture-in-hole tertile {i + 1}: recall {r[sel, 0].mean():.2f}" for i, sel in enumerate([t <= q[0], (t > q[0]) & (t <= q[1]), t > q[1]])]
    print(f"{m:10s} corr(recall, texture in hole) {np.corrcoef(r[:, 0], np.log1p(t))[0, 1]:+.2f} | corr(recall, hole size) "
          f"{np.corrcoef(r[:, 0], r[:, 3])[0, 1]:+.2f} | " + " | ".join(parts))
