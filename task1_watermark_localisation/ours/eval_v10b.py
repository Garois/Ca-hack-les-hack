"""PLAN_V10 W1 checks.
1. Variant selection itself inside nested CV (PixelSeal, TrustMark): per held fold, pick the variant whose nested
   per-image scores are best on the 4 other folds, score it on the held fold -> unbiased estimate of the whole procedure.
2. StegaStamp seeded-rectangle override: details of fired images, and acceptance threshold chosen inside nested CV
   among {0.8, 0.85, 0.9, 0.95}."""
import json
from pathlib import Path

import numpy as np

from common import load_masks128
from decision import boot
from eval_v10 import evaluate, guide_of, variants
from map_rules import is_full
from ps_probe import subtype
from rect_search import box_mask
from train_seg import folds
from wmloc.utils import miou_image

OUT = Path.home() / "wm_out"; gt = load_masks128()

for m in ("pixelseal", "trustmark"):
    items, fold_of = folds(m); names = [x["image_name"] for x in items]; G = {n: guide_of(n) for n in names}
    V = variants(m); ref = evaluate(m, V["v9:fx"], items, fold_of, G, use_gf=False)
    R = {(k, gf): evaluate(m, P, items, fold_of, G, use_gf=gf) for k, P in V.items() for gf in (False, True)}
    sel = {}
    for f in range(5):
        tr = [n for n in names if fold_of[n] != f]
        best = max(R, key=lambda k: np.mean([R[k][n] for n in tr]))
        for n in names:
            if fold_of[n] == f:
                sel[n] = R[best][n]
        print(f"   {m} fold {f}: selected {best}")
    mu, lo, hi = boot([sel[n] - ref[n] for n in names])
    print(f"== {m}: whole procedure (variant chosen inside nested CV) {np.mean(list(sel.values())):.4f} vs v9 "
          f"{np.mean(list(ref.values())):.4f} -> {mu:+.4f} (95% CI {lo:+.4f}..{hi:+.4f})", flush=True)

m = "stegastamp"
items, fold_of = folds(m); names = [x["image_name"] for x in items]; X = {x["image_name"]: x for x in items}
G = {n: guide_of(n) for n in names}; V = variants(m); P = V["v9:unet"]
rects = json.loads((OUT / "ssrect_train.json").read_text())
base = evaluate(m, P, items, fold_of, G, use_gf=False)
print("== stegastamp fired images (acc >= 0.8):")
for n, r in sorted(rects.items(), key=lambda kv: -kv[1]["acc"]):
    if r["acc"] >= 0.8:
        after = miou_image(box_mask(tuple(r["box"])), gt[n])
        print(f"   {n} acc {r['acc']:.2f} subtype {subtype(X[n], gt[n]):15s} mIoU {base[n]:.3f} -> {after:.3f}")
cand = (0.8, 0.85, 0.9, 0.95)


def with_thr(n, t):
    r = rects.get(n)
    if r is not None and not is_full(X[n]) and r["acc"] >= t:
        return miou_image(box_mask(tuple(r["box"])), gt[n])
    return base[n]


sel = {}
for f in range(5):
    tr = [n for n in names if fold_of[n] != f]
    t = max(cand, key=lambda t: np.mean([with_thr(n, t) for n in tr]))
    for n in names:
        if fold_of[n] == f:
            sel[n] = with_thr(n, t)
    print(f"   fold {f}: threshold {t}")
mu, lo, hi = boot([sel[n] - base[n] for n in names])
print(f"== stegastamp rect override, threshold chosen inside nested CV: {np.mean(list(sel.values())):.4f} vs "
      f"{np.mean(list(base.values())):.4f} -> {mu:+.4f} (95% CI {lo:+.4f}..{hi:+.4f})")
