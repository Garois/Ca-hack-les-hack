"""Exploration: shape of GT masks and whether decoding a crop around the GT region recovers the message."""
import numpy as np
import torch
from scipy import ndimage

from common import METHODS, bits, load_image, load_mask512, meta, wm

train = meta("train")
for method in METHODS:
    w = wm(method)
    for full in (False, True):
        xs = [x for x in train if x["method"] == method and x["full_watermark"] == full][:40]
        fills, acc_full, acc_crop, acc_clean = [], [], [], []
        for x in xs:
            img = load_image("train", x["image_name"]); m = load_mask512(x["image_name"]); msg = bits(x["message"])
            tgt = m if not full else 1 - m          # the small region: wm object (False) or clean hole (True)
            ys, xs_ = np.where(tgt)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs_.min(), xs_.max() + 1
            fills.append(tgt.sum() / ((y1 - y0) * (x1 - x0)))
            t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].float() / 255
            dec = lambda a: float((w.decode(t(a)).cpu().numpy()[0] == msg).mean())
            acc_full.append(dec(img))
            acc_crop.append(dec(img[y0:y1, x0:x1]))            # crop of the small region
            # crop of the largest clean/wm-complement rectangle: just the full image with the region greyed
            g = img.copy(); g[tgt.astype(bool)] = 128
            acc_clean.append(dec(g))
        print(f"{method:10s} full={full!s:5s} bboxfill={np.mean(fills):.2f} (>.95: {np.mean(np.array(fills) > .95):.2f})"
              f"  acc full-img={np.mean(acc_full):.3f}  acc crop(region)={np.mean(acc_crop):.3f}"
              f"  acc region-greyed={np.mean(acc_clean):.3f}", flush=True)
