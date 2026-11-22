"""Domain-gap diagnosis: compare watermark strength statistics between real train and synthetic images, per regime.
glob[:,0] = mean signed logit (whole image), glob[:,1] = bit accuracy; map = message map on wm vs clean cells."""
import numpy as np

from common import CACHE, METHODS, ROOT, load_masks128, meta

gt = load_masks128()
for method in METHODS:
    z = np.load(CACHE / f"feats_{method}.npz")
    z = {k: z[k] for k in z.files}                     # load once (npz re-reads the array on every access)
    idx = {n: i for i, n in enumerate(z["names"].tolist())}
    rows = {}
    for x in meta("train"):
        if x["method"] == method:
            rows.setdefault(("real", x["full_watermark"]), []).append((idx[x["image_name"]], gt[x["image_name"]]))
    for f in sorted((ROOT / "ours" / "synth" / method).glob("*.npz"))[:3000]:
        if f.stem in idx:
            s = np.load(f)
            rows.setdefault(("synth", bool(s["full"])), []).append((idx[f.stem], s["mask128"]))
    for k in sorted(rows):
        ii = [i for i, _ in rows[k]]; masks = [m for _, m in rows[k]]
        g = z["glob"][ii]
        wm = np.mean([z["maps"][i][m == 1].mean() for i, m in zip(ii, masks) if m.any()])
        cl = np.mean([z["maps"][i][m == 0].mean() for i, m in zip(ii, masks) if (m == 0).any()])
        frac = np.mean([m.mean() for m in masks])
        print(f"{method:10s} {k[0]:5s} full={k[1]!s:5s} n={len(ii):4d} frac={frac:.3f} "
              f"mlogit med={np.median(g[:, 0]):6.2f} bitacc med={np.median(g[:, 1]):.3f} "
              f"| map wm={wm:6.2f} clean={cl:6.2f}")
