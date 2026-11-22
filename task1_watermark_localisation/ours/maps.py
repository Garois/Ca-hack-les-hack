"""Step 2: message-aware localisation maps from the extractors' per-position logits.
map(x) = mean over bits of (2*msg-1) * per_position_logit  -> high where the known message is present.
Saves cache/maps_<method>.npz (name -> 128x128 float map) for train+test and prints the train mIoU of thresholded maps."""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from common import CACHE, bits, load_image, load_masks128, meta
from common import wm as load_wm
from wmloc.utils import miou_image


K = 4                                   # shifts per axis: 8x8 latent -> 32x32 map (shift-and-stitch)
SHIFTS = [a * 16 - 24 for a in range(K)]  # cell centres of the stitched map land on a regular 16 px grid


def translate(x, dy, dx, pad=32):
    """x: B x 3 x 512 x 512; content moved so that original pixel (y, x) appears at (y - dy, x - dx)."""
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return xp[:, :, pad + dy:pad + dy + x.shape[2], pad + dx:pad + dx + x.shape[3]]


def compute(method, items, split, bs=16):
    W = load_wm(method); out = {}
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        x = torch.stack([torch.from_numpy(load_image(split, it["image_name"])).permute(2, 0, 1).float() / 255
                         for it in chunk]).cuda()
        s = torch.stack([torch.from_numpy(bits(it["message"]).astype(np.float32) * 2 - 1) for it in chunk]).cuda()
        H = None
        for a, dy in enumerate(SHIFTS):
            for b, dx in enumerate(SHIFTS):
                pp = W.features(translate(x, dy, dx))["per_position_logits"].float()   # B x nbits x 8 x 8
                m = (pp * s[:, :, None, None]).mean(1)
                if H is None:
                    H = torch.zeros(len(chunk), m.shape[1] * K, m.shape[2] * K, device=m.device)
                H[:, a::K, b::K] = m
        m = F.interpolate(H[:, None], size=(128, 128), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
        for it, mm in zip(chunk, m):
            out[it["image_name"]] = mm
    return out, H.shape


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--method", default="pixelseal"); a = ap.parse_args()
    tr = [x for x in meta("train") if x["method"] == a.method]
    te = [x for x in meta("test") if x["method"] == a.method]
    mtr, shape = compute(a.method, tr, "train"); mte, _ = compute(a.method, te, "test")
    print(a.method, "per-position map shape", tuple(shape), flush=True)
    np.savez(CACHE / f"maps_{a.method}.npz", **{**mtr, **mte})
    gt = load_masks128()
    for full in (False, True):
        xs = [x for x in tr if x["full_watermark"] == full]
        vals = np.concatenate([mtr[x["image_name"]].ravel() for x in xs])
        lab = np.concatenate([gt[x["image_name"]].ravel() for x in xs])
        print(f"full={full}: map mean on wm cells {vals[lab == 1].mean():.3f} vs clean cells {vals[lab == 0].mean():.3f}")
    best = max(((np.mean([miou_image((mtr[x['image_name']] > t).astype(np.uint8), gt[x['image_name']]) for x in tr]), t)
                for t in np.linspace(-2, 6, 33)))
    print(f"global threshold: best train mIoU {best[0]:.4f} at t={best[1]:.2f}")
    # per-image adaptive: Otsu-like split of the map values
    sc = []
    for x in tr:
        m = mtr[x["image_name"]]; t = (m.max() + m.min()) / 2
        sc.append(miou_image((m > t).astype(np.uint8), gt[x["image_name"]]))
    print(f"per-image midrange threshold: train mIoU {np.mean(sc):.4f}")


if __name__ == "__main__":
    main()
