"""Decision-layer headroom: current (fixed threshold per regime) vs per-image ORACLE threshold vs ORACLE rectangle,
on the OOF probabilities of the v8 pipeline (PixelSeal probe, TrustMark 0.75 probe + 0.25 U-Net, StegaStamp U-Net)."""
import numpy as np
from pathlib import Path
from common import load_masks128
from map_rules import is_full
from oof import postproc, valprobs
from ps_probe import subtype
from train_seg import folds
from wmloc.utils import miou_image
PROBE = Path.home() / "wm_out" / "ps_probe"; gt = load_masks128()
def lp(p):
    d = {}
    for f in sorted(PROBE.glob(p)):
        z = np.load(f); d.update({k: z[k].astype(np.float32) for k in z.files})
    return d
CFG = {"pixelseal": ({True: (0.7, 16), False: (0.3, 16)}, 1.0, "heldprobs_fx*.npz"),
       "trustmark": ({True: (0.7, 0), False: (0.3, 16)}, 0.75, "heldprobs_tm_fx*.npz"),
       "stegastamp": ({True: (0.7, 16), False: (0.3, 0)}, 0.0, None)}
tot = {}
for m, (cfg, b, pat) in CFG.items():
    items, _ = folds(m); U = {}
    for f in range(5): U.update(valprobs(f"{m}_f{f}_s2"))
    P = lp(pat) if pat else {}
    rows = {}
    for x in items:
        n = x["image_name"]; g = gt[n]; full = is_full(x)
        p = b * P[n] + (1 - b) * U[n] if b > 0 else U[n]
        cur = miou_image(postproc(p, full, *cfg[full]), g)
        orc = max(miou_image((p >= t).astype(np.uint8), g) for t in np.unique(np.quantile(p, np.linspace(0, 1, 101))))
        rows.setdefault(subtype(x, g), []).append((cur, orc))
    print(f"== {m}")
    for k, v in sorted(rows.items()):
        v = np.array(v); print(f"   {k:15s} n={len(v):3d} current {v[:,0].mean():.3f} | oracle per-image threshold {v[:,1].mean():.3f} (+{v[:,1].mean()-v[:,0].mean():.3f})")
    a = np.array([r for v in rows.values() for r in v]); print(f"   ALL current {a[:,0].mean():.3f} -> oracle {a[:,1].mean():.3f}")
