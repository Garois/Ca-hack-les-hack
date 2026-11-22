"""Re-embedding correlation map (idea A of the plan), tested on REAL train images.
r' = embed(x, known msg) - x is the residual the embedder would add. Where x is already watermarked at full-image
scale, x contains ~r', so the local normalised correlation between the high-pass of x and the high-pass of r' is
positive; on clean pixels it is ~0. Reports, per regime and region shape, the mean map on wm vs clean cells and the
mean per-image AUC.
    GPU=0 run.sh exp_reembed.py pixelseal"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

from common import bits, load_image, load_masks128, meta
from common import wm as load_wm


def hp(t, k):
    return t - F.avg_pool2d(F.pad(t, (k // 2,) * 4, mode="reflect"), k, stride=1)


def corr_map(x, r, k=5, win=16):
    """x, r: 1 x 3 x 512 x 512 float. Local normalised correlation of high-passes, on the 128 grid."""
    a, b = hp(x, k), hp(r, k)
    pool = lambda t: F.avg_pool2d(F.pad(t.sum(1, keepdim=True), ((win - 4) // 2,) * 4, mode="reflect"), win, stride=4)
    return (pool(a * b) / torch.sqrt(pool(a * a) * pool(b * b) + 1e-12))[0, 0]


def auc(score, lab):
    pos, neg = score[lab == 1], score[lab == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = np.argsort(np.argsort(np.concatenate([pos, neg])))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))


method = sys.argv[1]; W = load_wm(method); gt = load_masks128()
rows = {}
for x in meta("train"):
    if x["method"] != method:
        continue
    img = load_image("train", x["image_name"]); msg = bits(x["message"])
    e = W.embed(img, msg)
    xt = torch.from_numpy(img).permute(2, 0, 1)[None].float().cuda() / 255
    rt = (torch.from_numpy(e).permute(2, 0, 1)[None].float().cuda() / 255) - xt
    g = gt[x["image_name"]]
    small = g if not x["full_watermark"] else 1 - g
    ys, xs = np.where(small)
    shape = "rect" if len(ys) and small.sum() / ((ys.ptp() + 1) * (xs.ptp() + 1)) > 0.95 else "object"
    for k in (3, 7):
        m = corr_map(xt, rt, k=k).cpu().numpy()[:128, :128]
        rows.setdefault((x["full_watermark"], shape, k), []).append((m[g == 1].mean() if g.any() else np.nan,
                                                                    m[g == 0].mean() if (g == 0).any() else np.nan,
                                                                    auc(m.ravel(), g.ravel())))
for key in sorted(rows):
    r = np.array(rows[key], dtype=float)
    print(f"{method:10s} full={key[0]!s:5s} {key[1]:6s} hp{key[2]} n={len(r):3d} wm={np.nanmean(r[:, 0]):.4f} "
          f"clean={np.nanmean(r[:, 1]):.4f} AUC={np.nanmean(r[:, 2]):.3f}", flush=True)
