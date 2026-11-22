"""Synthetic training data with the exact embedders.
    run.sh synth.py --method pixelseal --n 4000 --seed 0
Writes ours/synth/<method>/<id>.png (512x512 image) and <id>.npz (mask 512 uint8 + mask128) mimicking the two regimes:
  regime "partial" (full_watermark=False): a rectangle of a clean image is watermarked at its own scale; either the
      whole rectangle is kept (outpainting-like) or only an object inside it (transplantation onto a clean image).
  regime "full" (full_watermark=True): the whole image is watermarked, then an object-shaped region is restored to the
      clean original (object removal) or covered by a foreign clean object (object addition).
Area statistics follow the train split (small region ~0.3%..45% of the image, median ~6%)."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from common import ROOT, wm
from wmloc.utils import downsample_mask

COCO = ROOT.parent / "coco"
MAX_RATIO = {"pixelseal": 16, "stegastamp": 16, "trustmark": 2}


class Coco:
    def __init__(self):
        ann = json.loads((COCO / "annotations" / "instances_val2017.json").read_text())
        self.imgs = {i["id"]: i for i in ann["images"]}
        self.anns = [a for a in ann["annotations"] if not a["iscrowd"] and a["area"] > 400]
        self.by_img = {}
        for a in self.anns:
            self.by_img.setdefault(a["image_id"], []).append(a)
        self.ids = list(self.imgs)

    def image(self, rng, size=512):
        info = self.imgs[self.ids[rng.integers(len(self.ids))]]
        im = Image.open(COCO / "val2017" / info["file_name"]).convert("RGB")
        w, h = im.size; s = min(w, h)
        x0 = rng.integers(0, w - s + 1); y0 = rng.integers(0, h - s + 1)
        return np.asarray(im.crop((x0, y0, x0 + s, y0 + s)).resize((size, size), Image.Resampling.BICUBIC)), info, (x0, y0, s)

    def object_mask(self, rng):
        """Random COCO object mask cropped to its bbox (bool array)."""
        a = self.anns[rng.integers(len(self.anns))]
        info = self.imgs[a["image_id"]]
        rle = mask_utils.frPyObjects(a["segmentation"], info["height"], info["width"])
        m = mask_utils.decode(mask_utils.merge(rle)).astype(bool)
        ys, xs = np.where(m)
        return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1], a, info


def resize_mask(m, h, w):
    return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.Resampling.BILINEAR)) > 127


def sample_area(rng):
    return float(np.exp(rng.uniform(np.log(0.004), np.log(0.45))))


def sample_rect(rng, area_frac, max_ratio, size=512):
    r = float(np.exp(rng.uniform(0, np.log(max_ratio)))) if rng.random() < 0.8 else float(np.exp(rng.uniform(0, np.log(min(max_ratio, 1.5)))))
    if rng.random() < 0.5:
        r = 1 / r
    a = area_frac * size * size
    h = int(np.clip(np.sqrt(a / r), 24, size)); w = int(np.clip(a / h, 24, size))
    y = rng.integers(0, size - h + 1); x = rng.integers(0, size - w + 1)
    return y, x, h, w


def place_object(rng, coco, area_frac, size=512):
    """Object mask placed at a random position with the requested area (approx.)."""
    om, _, _ = coco.object_mask(rng)
    fill = om.mean()
    scale = np.sqrt(area_frac * size * size / max(fill * om.size, 1))
    h = int(np.clip(om.shape[0] * scale, 8, size)); w = int(np.clip(om.shape[1] * scale, 8, size))
    m = resize_mask(om, h, w)
    out = np.zeros((size, size), bool)
    y = rng.integers(0, size - h + 1); x = rng.integers(0, size - w + 1)
    out[y:y + h, x:x + w] = m
    return out


def gen_partial(rng, coco, W, method):
    img, _, _ = coco.image(rng)
    out = img.copy()
    y, x, h, w = sample_rect(rng, sample_area(rng), MAX_RATIO[method])
    msg = rng.integers(0, 2, W.nbits)
    if rng.random() < 0.5:                                    # outpainting-like: whole rectangle carries the watermark
        src = img[y:y + h, x:x + w] if rng.random() < 0.5 else coco.image(rng, 512)[0][:h, :w]
        out[y:y + h, x:x + w] = W.embed(np.ascontiguousarray(src), msg)
        mask = np.zeros((512, 512), bool); mask[y:y + h, x:x + w] = True
    else:                                                     # transplantation: object of a watermarked image
        # Real data (exp_shape.py): object-shaped regions carry the FULL-IMAGE-scale watermark at the SAME position
        # (moving the object breaks the pixel phase and kills the signal). So: watermark a whole 512 image, keep the
        # object in place, and paste it onto either a different clean image (transplant) or its own clean original.
        donor = coco.image(rng)[0] if rng.random() < 0.6 else img
        wd = W.embed(donor, msg)
        om, _, _ = coco.object_mask(rng)
        m = resize_mask(om, h, w)
        if m.sum() < 30:
            m[:] = True
        mask = np.zeros((512, 512), bool); mask[y:y + h, x:x + w] = m
        out[mask] = wd[mask]
    return out, mask, msg


def gen_full(rng, coco, W, method):
    img, _, _ = coco.image(rng)
    msg = rng.integers(0, 2, W.nbits)
    wimg = W.embed(img, msg)
    hole = np.zeros((512, 512), bool)
    for _ in range(1 if rng.random() < 0.7 else rng.integers(2, 4)):
        hole |= place_object(rng, coco, sample_area(rng) * 0.9)
    if hole.mean() > 0.49 or hole.sum() < 20:
        hole = place_object(rng, coco, 0.05)
    out = wimg.copy()
    if rng.random() < 0.6:
        out[hole] = img[hole]                                  # restore clean original (object removal)
    else:
        other = coco.image(rng)[0]
        out[hole] = other[hole]                                # foreign clean object (object addition)
    return out, ~hole, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--dir", default="synth2")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    coco = Coco(); W = wm(a.method)
    od = ROOT / "ours" / a.dir / a.method; od.mkdir(parents=True, exist_ok=True)
    # Shards of SHARD samples: <seed>_<k>.npy (images, N x 512 x 512 x 3 uint8, mmap-able) + <seed>_<k>_meta.npz.
    # The project's file quota is shared by all teams and nearly full: never write one file per sample.
    SHARD = 250
    buf = {"img": [], "m128": [], "full": [], "msg": []}

    def flush(k):
        names = [f"s{a.seed}_{k:03d}_{j:03d}" for j in range(len(buf["img"]))]
        tmp = od / f"s{a.seed}_{k:03d}.tmp.npy"
        np.save(tmp, np.stack(buf["img"]))
        np.savez(od / f"s{a.seed}_{k:03d}_meta.npz", names=np.array(names), mask128=np.stack(buf["m128"]),
                 full=np.array(buf["full"]), message=np.array(buf["msg"]))
        tmp.rename(od / f"s{a.seed}_{k:03d}.npy")
        for v in buf.values():
            v.clear()

    for i in range(a.n):
        regime = "full" if rng.random() < 0.5 else "partial"
        img, mask, msg = (gen_full if regime == "full" else gen_partial)(rng, coco, W, a.method)
        mask = mask.astype(np.uint8)
        buf["img"].append(img); buf["m128"].append(downsample_mask(mask)); buf["full"].append(regime == "full")
        buf["msg"].append("".join(map(str, msg)))
        if len(buf["img"]) == SHARD:
            flush(i // SHARD)
        if i % 500 == 0:
            print(a.method, i, regime, f"frac={mask.mean():.3f}", flush=True)
    if buf["img"]:
        flush(a.n // SHARD)
    print("done", a.method, a.n)


def iter_shards(method, synth_dir):
    """Yields (shard images memmap, meta dict) for every complete shard of a synthetic set."""
    d = ROOT / "ours" / synth_dir / method
    for f in sorted(d.glob("*_meta.npz")):
        img = d / f.name.replace("_meta.npz", ".npy")
        if img.exists():
            z = np.load(f)
            yield np.load(img, mmap_mode="r"), {k: z[k] for k in z.files}


if __name__ == "__main__":
    main()
