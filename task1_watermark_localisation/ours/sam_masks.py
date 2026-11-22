"""PLAN_V11 A (GPU): SAM ViT-B 'everything' segments for every PREDICTED-PARTIAL image (train: all predicted-partial
images, used out-of-fold; test: cache/partial_test_<m>.json). Masks are reduced to the official 128 grid
(downsample_mask), kept if 4 <= area < 0.6*N, and stored bit-packed:
-> ~/wm_out/sam/masks_<method>_<split>[_<shard>].npz  {name: uint8 array n x 2048 (np.packbits of 128x128)}"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.home() / "wm_out" / "pylib"))
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from common import CACHE, load_image, meta
from map_rules import is_full
from wmloc.utils import downsample_mask

OUT = Path.home() / "wm_out" / "sam"


def unpack(arr):
    return np.unpackbits(arr, axis=1)[:, :128 * 128].reshape(-1, 128, 128)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args(); m = a.method
    if a.split == "train":
        names = [x["image_name"] for x in meta("train") if x["method"] == m and not is_full(x)]
    else:
        names = json.loads((CACHE / f"partial_test_{m}.json").read_text())
    names = sorted(names)[a.shard::a.nshard]
    fout = OUT / (f"masks_{m}_{a.split}.npz" if a.nshard == 1 else f"masks_{m}_{a.split}_{a.shard}.npz")
    sam = sam_model_registry["vit_b"](checkpoint=str(OUT / "sam_vit_b.pth")).cuda().eval()
    gen = SamAutomaticMaskGenerator(sam, points_per_side=32, pred_iou_thresh=0.8, stability_score_thresh=0.85,
                                    min_mask_region_area=64)
    out = {}; t0 = time.time()
    for k, n in enumerate(names):
        with torch.no_grad():
            segs = gen.generate(load_image(a.split, n))
        ms = [downsample_mask(s["segmentation"].astype(np.uint8)) for s in segs]
        ms = [x for x in ms if 4 <= x.sum() < 0.6 * x.size]
        out[n] = np.packbits(np.stack(ms).reshape(len(ms), -1), axis=1) if ms else np.zeros((0, 2048), np.uint8)
        if k % 20 == 0:
            print(f"{m} {a.split} {k}/{len(names)} {time.time() - t0:.0f}s segments={len(ms)}", flush=True)
    np.savez_compressed(fout, **out)
    print("done", m, a.split, len(out), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
