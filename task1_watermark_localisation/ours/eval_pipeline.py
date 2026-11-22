"""Honest estimate of the submission pipeline on the fold-0 validation images (models never trained on them):
base predictor (as in predict.py) + exact-rectangle override for predicted-partial images. Also sweeps the
rectangle acceptance threshold."""
import json

import numpy as np

from common import CACHE, METHODS, load_masks128, meta
from global_rect import OUT
from map_rules import is_full, rule
from predict import RECT_THR
from rect_search import box_mask
from train_seg import CKPT, CKPT_HOME
from wmloc.utils import miou_image

gt = load_masks128(); M = {x["image_name"]: x for x in meta("train")}
R = json.loads((CACHE / "map_rules.json").read_text())


def vp(name):
    f = max([p for p in (CKPT / f"{name}_valprobs.npz", CKPT_HOME / f"{name}_valprobs.npz") if p.exists()],
            key=lambda p: p.stat().st_mtime)
    z = np.load(f); return dict(zip(z["names"].tolist(), z["probs"].astype(np.float32)))


BASE = {
    "pixelseal": ("maprule", None),
    "trustmark": ("avg", ["trustmark_f0_aux", "trustmark_f0_s2"]),
    "stegastamp": ("avg", ["stegastamp_f0_s2"]),
}
allscores = []
for m in METHODS:
    kind, ck = BASE[m]
    names = sorted(vp(f"{m}_f0_s2"))
    if kind == "maprule":
        maps = dict(np.load(CACHE / f"maps_{m}.npz"))
        pred = {n: rule(maps[n], is_full(M[n]), R[f"{m}/{is_full(M[n])}"]["q"], R[f"{m}/{is_full(M[n])}"]["smooth"]) for n in names}
    else:
        P = [vp(c) for c in ck]
        pred = {n: (sum(p[n] for p in P) / len(P) > 0.5).astype(np.uint8) for n in names}
    rf = [p for p in (CACHE / f"rect_{m}_val0.json", OUT / f"rect_{m}_val0.json") if p.exists()]
    rects = json.loads(rf[-1].read_text()) if rf else {}
    base = np.mean([miou_image(pred[n], gt[n]) for n in names])
    line = f"{m:10s} n={len(names)} rect results={len(rects)} base={base:.4f}"
    best = None
    for thr in sorted({0.65, 0.7, 0.75, 0.8, 0.85, RECT_THR[m]}):
        s = []; k = 0
        for n in names:
            r = rects.get(n)
            if r is not None and not is_full(M[n]) and r["acc"] >= thr:
                s.append(miou_image(box_mask(tuple(r["box"])), gt[n])); k += 1
            else:
                s.append(miou_image(pred[n], gt[n]))
        line += f" | thr{thr}: {np.mean(s):.4f} ({k} overrides)"
        if thr == RECT_THR[m]:
            best = s
    print(line)
    allscores += best
print(f"ALL (with RECT_THR) = {np.mean(allscores):.4f} over {len(allscores)} val images")
