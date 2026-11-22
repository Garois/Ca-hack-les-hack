"""Public unlabeled anchor photos (Imagenette 320px) -> uint8 [N,224,224,3] npz (resize short side 224, center crop).
Labels are discarded; anchors are only used to keep the model's outputs close to orig."""
import random
from pathlib import Path

import numpy as np
from PIL import Image

from common import HERE

root = HERE / "anchors" / "imagenette2-320"
rng = random.Random(0)


def load(split, n):
    files = sorted((root / split).rglob("*.JPEG"))
    rng.shuffle(files)
    out = []
    for f in files[:n]:
        im = Image.open(f).convert("RGB")
        w, h = im.size
        s = 224 / min(w, h)
        im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
        w, h = im.size
        l, t = (w - 224) // 2, (h - 224) // 2
        out.append(np.asarray(im.crop((l, t, l + 224, t + 224)), dtype=np.uint8))
    return np.stack(out)


tr = load("train", 6000)
va = load("val", 1500)
np.savez(HERE / "anchors" / "anchors.npz", train=tr, val=va)
print("anchors", tr.shape, va.shape)
