"""B4 prep (CPU): StegaStamp seed boxes from the v9 decision for every PREDICTED-PARTIAL image.
train: OOF TTA maps calibrated on the other folds (as in the nested evaluation); test: mean of the 6 calibrated
models (as in v9). Seed = bounding box (512 px) of the main region of the v9 decoded mask.
-> ~/wm_out/ss_seeds.json {"train": {name: [y0,y1,x0,x1]}, "test": {...}}"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from common import load_masks128, meta
from decision import apply_iso, fit_iso
from final_v9 import decode, test_models
from map_rules import is_full
from train_seg import folds
from tta_eval import maps as oof_maps

gt = load_masks128(); m = "stegastamp"; CFG = {True: (1, 16), False: (0, 0)}


def seed_of(mk):
    if not mk.any() or mk.mean() >= 0.5:
        return None
    lab, k = ndimage.label(mk); big = lab == (1 + np.argmax(ndimage.sum(mk, lab, range(1, k + 1))))
    ys, xs = np.where(big); return [int(ys.min() * 4), int(ys.max() * 4 + 4), int(xs.min() * 4), int(xs.max() * 4 + 4)]


items, fold_of = folds(m); P = oof_maps(m, True); names = [x["image_name"] for x in items]; X = {x["image_name"]: x for x in items}
fullp = {n: is_full(X[n]) for n in names}; seeds = {"train": {}, "test": {}}
for f in range(5):
    tr = [n for n in names if fold_of[n] != f and not fullp[n]]
    cal = fit_iso(np.concatenate([P[n].ravel() for n in tr]), np.concatenate([gt[n].ravel() for n in tr]).astype(float))
    for n in names:
        if fold_of[n] == f and not fullp[n]:
            s = seed_of(decode(apply_iso(cal, P[n]), *CFG[False]))
            if s:
                seeds["train"][n] = s
tr = [n for n in names if not fullp[n]]
cal = fit_iso(np.concatenate([P[n].ravel() for n in tr]), np.concatenate([gt[n].ravel() for n in tr]).astype(float))
TM = test_models(m)
for x in meta("test"):
    if x["method"] != m or is_full(x):
        continue
    n = x["image_name"]; q = np.mean([apply_iso(cal, mod[n]) for mod in TM], 0)
    s = seed_of(decode(q, *CFG[False]))
    if s:
        seeds["test"][n] = s
(Path.home() / "wm_out" / "ss_seeds.json").write_text(json.dumps(seeds))
print("seeds: train", len(seeds["train"]), "test", len(seeds["test"]))
