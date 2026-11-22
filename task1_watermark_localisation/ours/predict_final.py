"""Final submission: per method and predicted regime, the predictor chosen by oof.py (cache/final_config.json):
  "const" | "maprule" | "unet@<thr>/min<cells>" where the U-Net probability is the average of the 5 fold models and
  the all-train model (synth2, aux); then the exact-rectangle override (predict.RECT_THR).
    run.sh predict_final.py --out v6.pt"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from common import CACHE, METHODS, meta
from global_rect import OUT
from map_rules import is_full, rule
from oof import postproc
from predict import RECT_THR, load_rects, unet_probs
from rect_search import box_mask
from train_seg import CKPT, CKPT_HOME


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    cfg = json.loads((CACHE / "final_config.json").read_text()); R = json.loads((CACHE / "map_rules.json").read_text())
    pred, stats = {}, {}
    for m in METHODS:
        te = [x for x in meta("test") if x["method"] == m]
        models = [f"{m}_f{f}_s2" for f in range(5)] + [f"{m}_f-1_s2"]
        models = [c for c in models if (CKPT / f"{c}.pt").exists() or (CKPT_HOME / f"{c}.pt").exists()]
        P = None
        for c in models:
            p = unet_probs(c, "feats2", "test", te, m)
            P = p if P is None else {k: P[k] + p[k] for k in P}
        P = {k: v / len(models) for k, v in P.items()}
        maps = dict(np.load(CACHE / f"maps_{m}.npz")) if m != "stegastamp" else None
        rects = load_rects(m, "test")
        st = {"models": len(models), "rect": 0}
        for x in te:
            n = x["image_name"]; full = is_full(x); ch = cfg[f"{m}/{full}"]["choice"]
            if ch == "const":
                mk = np.full((128, 128), int(full), np.uint8)
            elif ch == "maprule":
                r = R[f"{m}/{full}"]; mk = rule(maps[n], full, r["q"], r["smooth"])
            else:
                thr, mc = ch[5:].split("/min"); mk = postproc(P[n], full, float(thr), int(mc))
            r = rects.get(n)
            if r is not None and not full and r["acc"] >= RECT_THR[m]:
                mk = box_mask(tuple(r["box"])); st["rect"] += 1
            pred[n] = mk
        st.update({f"choice_full={f}": cfg[f"{m}/{f}"]["choice"] for f in (False, True)})
        stats[m] = st
    names = [x["image_name"] for x in meta("test")]
    masks = torch.from_numpy(np.stack([pred[n] for n in names]).astype(bool))
    assert masks.shape == (1020, 128, 128) and len(set(names)) == 1020
    out = OUT / "submissions" / a.out; out.parent.mkdir(exist_ok=True)
    torch.save({"image_names": names, "masks": masks}, out)
    chk = torch.load(out, weights_only=True); assert chk["image_names"] == names and chk["masks"].shape == (1020, 128, 128)
    print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB", json.dumps(stats))
    try:
        dst = Path(__file__).parent / "submissions" / a.out; shutil.copy(out, dst); print("copied to", dst)
    except OSError as e:
        print("WARNING: copy failed:", e)


if __name__ == "__main__":
    main()
