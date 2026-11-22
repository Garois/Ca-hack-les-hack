"""Shared helpers: paths, data loading, watermark models, official evaluation."""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wmloc.utils import miou_image, score  # noqa: E402

DATA = ROOT / "release"
CACHE = ROOT / "ours" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
METHODS = ("pixelseal", "trustmark", "stegastamp")


def meta(split):
    return json.loads((DATA / split / "metadata.json").read_text())


def load_image(split, name) -> np.ndarray:
    return np.asarray(Image.open(DATA / split / "images" / f"{name}.png").convert("RGB"))


def load_masks128():
    z = np.load(DATA / "train" / "masks_128.npz")
    return dict(zip(z["names"].tolist(), z["masks"]))


def load_mask512(name):
    return (np.asarray(Image.open(DATA / "train" / "masks" / f"{name}.png")) > 127).astype(np.uint8)


def bits(msg: str) -> np.ndarray:
    return np.array([int(c) for c in msg], dtype=np.uint8)


_WM = {}


def wm(method, device="cuda"):
    if method not in _WM:
        from wmloc.watermarks import load_watermark
        _WM[method] = load_watermark(method, ROOT / "models", device)
    return _WM[method]


def report(preds: dict, title=""):
    """preds: name -> 128x128 0/1. Prints official mIoU on train, global / per method / per regime."""
    gt = load_masks128()
    groups = collections.defaultdict(list)
    for x in meta("train"):
        n = x["image_name"]
        if n not in preds:
            continue
        s = miou_image(preds[n], gt[n])
        groups["ALL"].append(s)
        groups[x["method"]].append(s)
        groups[f"{x['method']}/full={x['full_watermark']}"].append(s)
    line = " ".join(f"{k}={np.mean(v):.4f}" for k, v in sorted(groups.items()))
    print(f"[{title}] {line}", flush=True)
    return float(np.mean(groups["ALL"]))
