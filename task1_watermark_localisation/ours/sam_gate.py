"""Reflection 3 gate (run inside the Slurm job, GPU): does SAM segment voting improve TRANSPLANTED OBJECTS (visible contours)?
For partial-object train images: v9 calibrated OOF map q -> (a) v9 decoded mask, (b) union of SAM 'everything' segments
whose mean q >= tau (fallback to (a) if none). Reported for several tau, plus the oracle best union of segments."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.home() / "wm_out" / "pylib"))
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from common import load_image, load_masks128
from decision import apply_iso, clean_full_or_partial, emiou_mask, fit_iso
from map_rules import is_full
from ps_probe import subtype
from train_seg import folds
from tta_eval import maps as oof_maps
from wmloc.utils import downsample_mask, miou_image
from scipy import ndimage

torch.set_num_threads(16); gt = load_masks128()
dev = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint=str(Path.home() / "wm_out" / "sam" / "sam_vit_b.pth")).to(dev).eval()
gen = SamAutomaticMaskGenerator(sam, points_per_side=16, pred_iou_thresh=0.8, stability_score_thresh=0.85, min_mask_region_area=64)
TAUS = (0.3, 0.4, 0.5, 0.6)
res = {}
for m, k in (("pixelseal", 5), ("trustmark", 5), ("stegastamp", 4)):
    items, fold_of = folds(m); P = oof_maps(m, True); names = [x["image_name"] for x in items]; X = {x["image_name"]: x for x in items}
    fullp = {n: is_full(X[n]) for n in names}; done = 0
    for f in range(5):
        tr = [n for n in names if fold_of[n] != f and not fullp[n]]
        cal = fit_iso(np.concatenate([P[n].ravel() for n in tr]), np.concatenate([gt[n].ravel() for n in tr]).astype(float))
        for n in names:
            if fold_of[n] != f or fullp[n] or subtype(X[n], gt[n]) != "partial/object" or done >= k:
                continue
            q = apply_iso(cal, P[n]); base = clean_full_or_partial(emiou_mask(q), 0)
            segs = [downsample_mask(s["segmentation"].astype(np.uint8)) for s in gen.generate(load_image("train", n))]
            segs = [s for s in segs if 0 < s.sum() < 0.6 * s.size]
            row = {"base": miou_image(base, gt[n])}
            for t in TAUS:
                sel = [s for s in segs if q[s == 1].mean() >= t]
                mk = np.clip(np.sum(sel, 0), 0, 1).astype(np.uint8) if sel else base
                row[t] = miou_image(mk, gt[n])
            # oracle: segments that are >= 50% inside GT
            sel = [s for s in segs if gt[n][s == 1].mean() >= 0.5]
            row["oracle"] = miou_image(np.clip(np.sum(sel, 0), 0, 1).astype(np.uint8), gt[n]) if sel else row["base"]
            res.setdefault(m, []).append(row); done += 1
            print(m, n, {kk: round(v, 3) for kk, v in row.items()}, f"segments {len(segs)}", flush=True)
for m, rows in res.items():
    print(f"== {m} n={len(rows)}: base {np.mean([r['base'] for r in rows]):.3f} | " +
          " ".join(f"tau{t}: {np.mean([r[t] for r in rows]):.3f}" for t in TAUS) + f" | oracle {np.mean([r['oracle'] for r in rows]):.3f}")
