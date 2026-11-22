"""Other factors: (i) number of connected components of the GT region in partial images, per method/subtype;
(ii) visibility of rectangle contours (gradient on the contour vs shifted contour), as done for objects."""
import numpy as np
from scipy import ndimage
from common import load_image, load_mask512, load_masks128, meta
from exp_edges import ratio
from ps_probe import subtype
gt = load_masks128(); rng = np.random.default_rng(0)
for m in ("pixelseal", "trustmark", "stegastamp"):
    rows = {}
    for x in meta("train"):
        if x["method"] != m or x["full_watermark"]:
            continue
        g = gt[x["image_name"]]; st = subtype(x, g); lab, k = ndimage.label(g)
        sizes = sorted(ndimage.sum(g, lab, range(1, k + 1)), reverse=True) if k else []
        big = [s for s in sizes if s >= 0.1 * sizes[0]] if sizes else []
        rr = ratio(load_image("train", x["image_name"]), load_mask512(x["image_name"]), rng) if st == "partial/rect" else np.nan
        rows.setdefault(st, []).append((k, len(big), (sizes[1] / sizes[0]) if len(sizes) > 1 else 0.0, rr))
    for st, v in sorted(rows.items()):
        v = np.array(v, dtype=float)
        print(f"{m:10s} {st:15s} n={len(v):2d} components: >=2 in {np.mean(v[:,0] >= 2):.2f}, >=2 sizeable (>=10% of largest) in {np.mean(v[:,1] >= 2):.2f} | "
              f"2nd/1st size median {np.median(v[:,2]):.2f}" + (f" | rect contour edge ratio median {np.nanmedian(v[:,3]):.2f}, >1.3 in {np.nanmean(v[:,3] > 1.3):.2f}" if st == "partial/rect" else ""))
