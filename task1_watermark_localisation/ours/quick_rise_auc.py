"""Quick read: AUC and oracle mIoU of RISE maps vs the S4 message map on the images available in a RISE file."""
import sys
import numpy as np
from common import CACHE, load_masks128, meta
from exp_ps_maps import auc
from wmloc.utils import miou_image
z = np.load(sys.argv[1]); gt = load_masks128(); S4 = dict(np.load(CACHE / "maps_pixelseal.npz"))
F = {x["image_name"]: x["full_watermark"] for x in meta("train")}
rows = []
for n in z.files:
    g = gt[n]; w = z[n]
    orc = lambda m: max(miou_image((m >= t).astype(np.uint8), g) for t in np.percentile(m, np.arange(1, 100, 2)))
    rows.append((F[n], auc(w.ravel(), g.ravel()), auc(S4[n].ravel(), g.ravel()), orc(w), orc(S4[n].astype(np.float32))))
r = np.array(rows, dtype=float)
for full in (1.0, 0.0):
    s = r[r[:, 0] == full]
    if len(s):
        print(f"full={bool(full)} n={len(s)}  AUC rise={np.nanmean(s[:,1]):.3f} vs S4={np.nanmean(s[:,2]):.3f} | oracle mIoU rise={s[:,3].mean():.4f} vs S4={s[:,4].mean():.4f}")
