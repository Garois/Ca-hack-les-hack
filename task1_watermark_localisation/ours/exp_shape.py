"""Real partial images split by region shape (rectangle vs object): message-map contrast per group."""
import numpy as np
from common import CACHE, METHODS, load_masks128, meta
gt = load_masks128()
for method in METHODS:
    z = np.load(CACHE / f"feats_{method}.npz"); idx = {n: i for i, n in enumerate(z["names"].tolist())}; maps = z["maps"]; glob = z["glob"]
    for full in (False, True):
        grp = {"rect": [], "object": []}
        for x in meta("train"):
            if x["method"] != method or x["full_watermark"] != full: continue
            g = gt[x["image_name"]]; t = g if not full else 1 - g
            ys, xs = np.where(t)
            if len(ys) == 0: continue
            fill = t.sum() / ((ys.ptp() + 1) * (xs.ptp() + 1))
            grp["rect" if fill > 0.95 else "object"].append((maps[idx[x["image_name"]]].astype(np.float32), g, glob[idx[x["image_name"]], 1]))
        for k, v in grp.items():
            if not v: continue
            wm = np.mean([m[g == 1].mean() for m, g, _ in v if g.any()]); cl = np.mean([m[g == 0].mean() for m, g, _ in v if (g == 0).any()])
            print(f"{method:10s} full={full!s:5s} {k:6s} n={len(v):3d} map wm={wm:6.2f} clean={cl:6.2f} bitacc med={np.median([b for *_, b in v]):.3f}")
