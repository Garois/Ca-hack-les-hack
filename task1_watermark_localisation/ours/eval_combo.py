"""Fair comparison on the fold-0 validation images (never trained on) of the base predictors:
U-Net synth v1 (aux), U-Net synth2, their probability average, and the map rule (PixelSeal/TrustMark)."""
import json

import numpy as np

from common import CACHE, ROOT, load_masks128, meta
from map_rules import is_full, rule
from wmloc.utils import miou_image

gt = load_masks128(); M = {x["image_name"]: x for x in meta("train")}
R = json.loads((CACHE / "map_rules.json").read_text())


def vp(name):
    from train_seg import CKPT, CKPT_HOME
    f = max([p for p in (CKPT / f"{name}_valprobs.npz", CKPT_HOME / f"{name}_valprobs.npz") if p.exists()], key=lambda p: p.stat().st_mtime)
    z = np.load(f)
    return dict(zip(z["names"].tolist(), z["probs"].astype(np.float32)))


for m in ("pixelseal", "trustmark", "stegastamp"):
    a, b = vp(f"{m}_f0_aux"), vp(f"{m}_f0_s2")
    names = sorted(a)
    res = {}
    for label, f in [("unet_v1", lambda n: a[n]), ("unet_s2", lambda n: b[n]), ("avg", lambda n: (a[n] + b[n]) / 2)]:
        for t in (0.4, 0.5, 0.6):
            res[f"{label}@{t}"] = np.mean([miou_image((f(n) > t).astype(np.uint8), gt[n]) for n in names])
    if m != "stegastamp":
        maps = dict(np.load(CACHE / f"maps_{m}.npz"))
        res["maprule"] = np.mean([miou_image(rule(maps[n], is_full(M[n]), R[f"{m}/{is_full(M[n])}"]["q"],
                                                  R[f"{m}/{is_full(M[n])}"]["smooth"]), gt[n]) for n in names])
    res["regime_const"] = np.mean([miou_image(np.full((128, 128), int(is_full(M[n])), np.uint8), gt[n]) for n in names])
    print(m, f"n={len(names)}", " ".join(f"{k}={v:.4f}" for k, v in res.items()))
