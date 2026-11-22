"""Which generation process reproduces the REAL partial-regime statistics?
Real train (full_watermark=False): message map inside the wm region vs outside
  pixelseal 0.56 vs 0.26, trustmark 0.25 vs 0.05 (gap.py); our synth gives ~0.01 vs 0.00.
Variants for a region R (COCO object inside a rectangle, or the rectangle itself):
  A  inplace : whole 512 image watermarked, R kept, everything else restored to the clean original
  B  moved   : donor watermarked as a whole, R cut from the donor and pasted at another position of a clean image
  C  cropscale: only the rectangle is watermarked at its own scale (crop -> embed -> paste back), R kept
  D  bigrect : a large rectangle (2-4x the region) is watermarked at its own scale, R kept
    GPU=0 run.sh exp_partial.py --method pixelseal --n 120"""
import argparse

import numpy as np
import torch

from common import bits
from common import wm as load_wm
from feats import batch_feats
from synth import MAX_RATIO, Coco, resize_mask, sample_area, sample_rect
from wmloc.utils import downsample_mask


def region(rng, coco, method):
    y, x, h, w = sample_rect(rng, sample_area(rng), MAX_RATIO[method])
    R = np.zeros((512, 512), bool)
    if rng.random() < 0.5:
        R[y:y + h, x:x + w] = True
    else:
        m = resize_mask(coco.object_mask(rng)[0], h, w)
        R[y:y + h, x:x + w] = m if m.sum() > 30 else True
    return R, (y, x, h, w)


def make(variant, rng, coco, W, method):
    img = coco.image(rng)[0]; msg = rng.integers(0, 2, W.nbits)
    R, (y, x, h, w) = region(rng, coco, method)
    out = img.copy()
    if variant == "A":
        wi = W.embed(img, msg); out[R] = wi[R]
    elif variant == "B":
        donor = coco.image(rng)[0]; wd = W.embed(donor, msg)
        dy, dx = rng.integers(0, 512 - h + 1), rng.integers(0, 512 - w + 1)
        sub = R[y:y + h, x:x + w]
        out[y:y + h, x:x + w][sub] = wd[dy:dy + h, dx:dx + w][sub]
    elif variant == "C":
        wc = W.embed(np.ascontiguousarray(img[y:y + h, x:x + w]), msg)
        sub = R[y:y + h, x:x + w]; out[y:y + h, x:x + w][sub] = wc[sub]
    elif variant == "D":
        s = rng.uniform(1.5, 2.0)
        H = int(min(512, h * s)); Wd = int(min(512, w * s))
        y0 = int(np.clip(y - (H - h) // 2, 0, 512 - H)); x0 = int(np.clip(x - (Wd - w) // 2, 0, 512 - Wd))
        wc = img.copy(); wc[y0:y0 + H, x0:x0 + Wd] = W.embed(np.ascontiguousarray(img[y0:y0 + H, x0:x0 + Wd]), msg)
        out[R] = wc[R]
    return out, R, msg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--method", default="pixelseal"); ap.add_argument("--n", type=int, default=120)
    a = ap.parse_args()
    coco = Coco(); W = load_wm(a.method)
    for variant in "ABCD":
        rng = np.random.default_rng(7)
        imgs, Rs, msgs = [], [], []
        for _ in range(a.n):
            o, R, m = make(variant, rng, coco, W, a.method); imgs.append(o); Rs.append(downsample_mask(R.astype(np.uint8))); msgs.append(m)
        maps, glob = [], []
        for i in range(0, a.n, 24):
            mp, g = batch_feats(W, a.method, imgs[i:i + 24], msgs[i:i + 24]); maps.append(mp); glob.append(g)
        maps = np.concatenate(maps).astype(np.float32); glob = np.concatenate(glob)
        wm_ = np.mean([m[r == 1].mean() for m, r in zip(maps, Rs) if r.any()])
        cl_ = np.mean([m[r == 0].mean() for m, r in zip(maps, Rs)])
        print(f"{a.method:10s} variant {variant}: map wm={wm_:6.2f} clean={cl_:6.2f}  bitacc med={np.median(glob[:, 1]):.3f}"
              f"  frac={np.mean([r.mean() for r in Rs]):.3f}", flush=True)


if __name__ == "__main__":
    main()
