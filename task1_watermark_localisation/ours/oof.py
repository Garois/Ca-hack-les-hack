"""Out-of-fold analysis over the whole train split (5 folds of the synth2 U-Nets, never trained on their val fold).
Per method and per PREDICTED regime, picks the best of: U-Net OOF probabilities at a threshold (optionally with
small-component removal), the map rule, or the constant regime mask. Writes cache/final_config.json.
(The map rule's parameters were fitted on all of train: its numbers are slightly optimistic.)"""
import json

import numpy as np
from scipy import ndimage

from common import CACHE, METHODS, load_masks128, meta
from map_rules import is_full, rule
from train_seg import CKPT, CKPT_HOME
from wmloc.utils import miou_image

gt = load_masks128(); R = json.loads((CACHE / "map_rules.json").read_text())


def valprobs(name):
    c = [p for p in (CKPT / f"{name}_valprobs.npz", CKPT_HOME / f"{name}_valprobs.npz") if p.exists()]
    if not c:
        return {}
    z = np.load(max(c, key=lambda p: p.stat().st_mtime)); return dict(zip(z["names"].tolist(), z["probs"].astype(np.float32)))


def clean(mask, min_cells):
    if min_cells <= 0:
        return mask
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    keep = np.isin(lab, 1 + np.flatnonzero(sizes >= min_cells))
    return keep.astype(np.uint8)


def postproc(p, full, thr, min_cells):
    m = (p > thr).astype(np.uint8)
    if full:      # mostly watermarked: clean the predicted holes (class 0 components)
        return 1 - clean(1 - m, min_cells)
    return clean(m, min_cells)


def main():
    config = {}
    for m in METHODS:
        P = {}
        for f in range(5):
            P.update(valprobs(f"{m}_f{f}_s2"))
        items = [x for x in meta("train") if x["method"] == m and x["image_name"] in P]
        maps = dict(np.load(CACHE / f"maps_{m}.npz")) if m != "stegastamp" else None
        print(f"== {m}: OOF probabilities for {len(items)}/170 images", flush=True)
        for full in (False, True):
            grp = [x for x in items if is_full(x) == full]
            if not grp:
                continue
            cands = {}
            cands["const"] = np.mean([miou_image(np.full((128, 128), int(full), np.uint8), gt[x["image_name"]]) for x in grp])
            if maps is not None:
                r = R[f"{m}/{full}"]
                cands["maprule"] = np.mean([miou_image(rule(maps[x["image_name"]], full, r["q"], r["smooth"]), gt[x["image_name"]]) for x in grp])
            for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
                for mc in (0, 4, 16):
                    cands[f"unet@{thr}/min{mc}"] = np.mean([miou_image(postproc(P[x["image_name"]], full, thr, mc), gt[x["image_name"]]) for x in grp])
            best = max(cands, key=cands.get)
            top = sorted(cands.items(), key=lambda kv: -kv[1])[:4]
            print(f"   pred_full={full!s:5s} n={len(grp):3d} const={cands['const']:.4f} maprule={cands.get('maprule', float('nan')):.4f} "
                  f"unet@0.5={cands['unet@0.5/min0']:.4f} | best {best}={cands[best]:.4f} | top: " + ", ".join(f"{k}={v:.4f}" for k, v in top))
            config[f"{m}/{full}"] = {"choice": best, "score": float(cands[best]), "n": len(grp)}
    (CACHE / "final_config.json").write_text(json.dumps(config, indent=1))
    print("wrote", CACHE / "final_config.json")


if __name__ == "__main__":
    main()
