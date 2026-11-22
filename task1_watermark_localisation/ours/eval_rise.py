"""Evaluate the RISE occlusion map for PixelSeal on train, honestly (5-fold OOF tuning of the rule parameters).
Predicted-full images: hole = smoothed (w / median w) < tau  (optionally mixed with the S4 message map).
Predicted-partial images: object = smoothed w / max > tau.
Compared with the current map rule (map_rules.py) on the same images.
    run.sh eval_rise.py --cell 32"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from common import CACHE, load_masks128, meta
from map_rules import is_full, rule as cur_rule
from oof import clean
from train_seg import folds
from wmloc.utils import miou_image

RISE = Path.home() / "wm_out" / "rise"


def load_rise(split, cell):
    d = {}
    for f in sorted(RISE.glob(f"rise_{split}_c{cell}_*.npz")):
        z = np.load(f); d.update({k: z[k] for k in z.files})
    return d


def zs(m):
    return (m - np.median(m)) / (np.percentile(m, 90) - np.percentile(m, 10) + 1e-9)


def pred_full(w, s4, tau, sm, mix, minc):
    if mix > 0:
        v = (1 - mix) * zs(w) + mix * zs(s4)
    else:
        v = w / (np.median(w) + 1e-9)
        v = (v - 1)                                                   # 0 at the watermarked level
    v = ndimage.gaussian_filter(v, sm) if sm else v
    hole = (v < tau).astype(np.uint8)
    return 1 - clean(hole, minc)


def pred_partial(w, tau, sm, minc):
    v = ndimage.gaussian_filter(w, sm) if sm else w
    mx = v.max()
    if mx <= 0:
        return np.zeros_like(v, np.uint8)
    return clean((v >= tau * mx).astype(np.uint8), minc)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cell", type=int, default=32); a = ap.parse_args()
    W = load_rise("train", a.cell); gt = load_masks128()
    S4 = dict(np.load(CACHE / "maps_pixelseal.npz")); R = json.loads((CACHE / "map_rules.json").read_text())
    items, fold_of = folds("pixelseal")
    items = [x for x in items if x["image_name"] in W]
    print(f"RISE maps for {len(items)}/170 train images (cell {a.cell})")
    grids = {
        True: [dict(tau=t, sm=sm, mix=mx, minc=mc) for t in (-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1) for sm in (0, 2, 4)
               for mx in (0.0, 0.3, 0.5) for mc in (0, 16)],
        False: [dict(tau=t, sm=sm, minc=mc) for t in (0.3, 0.4, 0.5, 0.6, 0.7) for sm in (0, 2, 4) for mc in (0, 16)],
    }
    for full in (True, False):
        grp = [x for x in items if is_full(x) == full]
        r = R[f"pixelseal/{full}"]
        cur = {x["image_name"]: miou_image(cur_rule(S4[x["image_name"]], full, r["q"], r["smooth"]), gt[x["image_name"]]) for x in grp}

        def score(p, x):
            n = x["image_name"]
            if full:
                return miou_image(pred_full(W[n], S4[n].astype(np.float32), **p), gt[n])
            return miou_image(pred_partial(W[n], **p), gt[n])

        table = {i: {x["image_name"]: score(p, x) for x in grp} for i, p in enumerate(grids[full])}
        oof = []
        for f in range(5):
            tr = [x["image_name"] for x in grp if fold_of[x["image_name"]] != f]
            va = [x["image_name"] for x in grp if fold_of[x["image_name"]] == f]
            best = max(table, key=lambda i: np.mean([table[i][n] for n in tr]))
            oof += [table[best][n] for n in va]
        allbest = max(table, key=lambda i: np.mean(list(table[i].values())))
        auc_note = ""
        print(f"pred_full={full!s:5s} n={len(grp):3d} current rule={np.mean(list(cur.values())):.4f} | RISE OOF={np.mean(oof):.4f} "
              f"| in-sample best={np.mean(list(table[allbest].values())):.4f} with {grids[full][allbest]}{auc_note}")
        if full:
            Path(CACHE / "rise_rule.json").write_text(json.dumps({"full": grids[full][allbest]}))


if __name__ == "__main__":
    main()
