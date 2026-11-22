"""Regime-aware decision rules on the cached message maps (no training). Grid-searches, per method and per regime,
a threshold relative to the image's own map statistics, and reports the train mIoU."""
import json

import numpy as np
from scipy import ndimage

from common import CACHE, METHODS, load_masks128, meta
from wmloc.utils import miou_image

thr = json.loads((CACHE / "regime_thresholds.json").read_text())
z = np.load(CACHE / "global_logits.npz", allow_pickle=True); G = {n: v for n, v in zip(z["names"], z["vals"])}
gt = load_masks128()


def is_full(x):
    return float((G[x["image_name"]] > 0).mean()) >= thr[x["method"]]


def rule(m, full, q, smooth):
    if smooth:
        m = ndimage.gaussian_filter(m, smooth)
    if full:   # mostly watermarked: clean hole where the map drops below a fraction of the image's high level
        hi = np.percentile(m, 90)
        return (m >= q * hi).astype(np.uint8)
    hi = m.max()  # mostly clean: watermarked region where the map exceeds a fraction of its peak
    return (m >= q * hi).astype(np.uint8) if hi > 0.5 else np.zeros_like(m, np.uint8)


best_rules = {}
for method in ("pixelseal", "trustmark"):
    maps = dict(np.load(CACHE / f"maps_{method}.npz"))
    xs = [x for x in meta("train") if x["method"] == method]
    for full in (False, True):
        grp = [x for x in xs if is_full(x) == full]
        base = np.mean([miou_image(np.full((128, 128), int(full), np.uint8), gt[x["image_name"]]) for x in grp])
        res = []
        for q in np.linspace(0.05, 0.95, 19):
            for sm in (0, 2, 4):
                res.append((np.mean([miou_image(rule(maps[x["image_name"]], full, q, sm), gt[x["image_name"]]) for x in grp]), q, sm))
        b = max(res)
        best_rules[f"{method}/{full}"] = {"q": float(b[1]), "smooth": b[2], "score": float(b[0]), "const": float(base)}
        print(f"{method:10s} pred_full={full!s:5s} n={len(grp)} constant={base:.4f}  best rule={b[0]:.4f} (q={b[1]:.2f}, smooth={b[2]})")
(CACHE / "map_rules.json").write_text(json.dumps(best_rules, indent=1))
