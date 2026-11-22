"""Do GT holes / transplanted objects follow the image's own object contours?
Compares the image gradient magnitude on the GT mask boundary with the same boundary shifted by 24 px
(same shape, random place). Ratio >> 1: masks follow real image contours. Also on synth2 (random shapes) for contrast."""
import numpy as np
from scipy import ndimage

from common import load_image, load_mask512, meta
from synth import iter_shards


def grad_mag(img):
    g = img.astype(np.float32).mean(2)
    return np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1))


def boundary(m):
    return m ^ ndimage.binary_erosion(m, iterations=1)


def ratio(img, m512, rng):
    gm = grad_mag(img); b = boundary(m512.astype(bool))
    if b.sum() < 20:
        return np.nan
    on = gm[b].mean(); offs = []
    for _ in range(4):
        dy, dx = rng.integers(-24, 25, 2)
        bs = np.roll(np.roll(b, dy, 0), dx, 1); offs.append(gm[bs].mean())
    return on / (np.mean(offs) + 1e-6)


rng = np.random.default_rng(0)
for method in ("pixelseal", "trustmark", "stegastamp"):
    rows = {}
    for x in meta("train"):
        if x["method"] != method:
            continue
        m = load_mask512(x["image_name"]); small = m if not x["full_watermark"] else 1 - m
        ys, xs = np.where(small)
        shape = "rect" if len(ys) and small.sum() / ((ys.ptp() + 1) * (xs.ptp() + 1)) > 0.95 else "object"
        if shape == "rect":
            continue
        k = "full/hole" if x["full_watermark"] else "partial/object"
        rows.setdefault(k, []).append(ratio(load_image("train", x["image_name"]), small, rng))
    for k, v in rows.items():
        v = np.array(v, dtype=float)
        print(f"REAL  {method:10s} {k:15s} n={len(v):3d} edge ratio median {np.nanmedian(v):.2f}  frac>1.3: {np.nanmean(v > 1.3):.2f}")
imgs, mt = next(iter_shards("pixelseal", "synth2"))
v = {"full/hole": [], "partial/object": []}
for j in range(len(mt["names"])):
    m128 = mt["mask128"][j]; full = bool(mt["full"][j])
    m512 = np.kron(m128, np.ones((4, 4), np.uint8)); small = m512 if not full else 1 - m512
    ys, xs = np.where(small)
    if not len(ys) or (not full and small.sum() / ((ys.ptp() + 1) * (xs.ptp() + 1)) > 0.95):
        continue
    v["full/hole" if full else "partial/object"].append(ratio(np.asarray(imgs[j]), small, rng))
for k, vv in v.items():
    vv = np.array(vv, dtype=float)
    print(f"SYNTH pixelseal  {k:15s} n={len(vv):3d} edge ratio median {np.nanmedian(vv):.2f}  frac>1.3: {np.nanmean(vv > 1.3):.2f}")
