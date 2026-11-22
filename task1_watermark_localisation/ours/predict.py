"""Assemble a submission. Per method, a base predictor spec, then (optionally) the exact-rectangle override.
Base specs:
    regime                                     all-1 / all-0 from the regime classifier
    maprule                                    regime-aware threshold of the message map (map_rules.py)
    unet:<ckpt>[:<feats>][@thr]                U-Net checkpoint; with <feats> it is an aux model using that feature file
    avg:<ckpt>:<feats>+<ckpt>:<feats>[@thr]    probability average of several (aux) U-Nets
Rect override (--rect): for images of the partial regime whose global rectangle search (global_rect.py) reached a
bit accuracy >= RECT_THR[method], the mask is that rectangle.
    run.sh predict.py --out v4.pt --rect --pixelseal maprule \
        --trustmark "avg:trustmark_f0_aux:feats+trustmark_f0_s2:feats2" --stegastamp "unet:stegastamp_f0_s2:feats2"
Writes to ~/wm_out/submissions/ (home quota) and tries to copy into ours/submissions/."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from common import CACHE, METHODS, meta, report
from global_rect import OUT
from map_rules import is_full, rule
from rect_search import box_mask
from train_seg import N_AUX, Feats, UNet, ckpt_path, load_split, predict_probs

RECT_THR = {"pixelseal": 0.80, "trustmark": 0.80, "stegastamp": 0.90}   # stegastamp: 4 px misalignment kills decoding, global search only finds noise


def unet_probs(ckpt, feats_name, split, items, method):
    aux = bool(feats_name)
    model = UNet(pretrained=False, n_aux=N_AUX if aux else 0).cuda()
    model.load_state_dict(torch.load(ckpt_path(ckpt), map_location="cuda", weights_only=True))
    imgs, a = load_split(split, items, Feats(method, feats_name) if aux else None)
    return dict(zip([x["image_name"] for x in items], predict_probs(model, imgs, a).numpy()))


def parse_models(body):
    out = []
    for part in body.split("+"):
        f = part.split(":")
        out.append((f[0], f[1] if len(f) > 1 else ""))
    return out


def base(spec, method, split, items):
    if spec == "regime":
        return {x["image_name"]: np.full((128, 128), int(is_full(x)), np.uint8) for x in items}
    if spec == "maprule":
        maps = dict(np.load(CACHE / f"maps_{method}.npz")); R = json.loads((CACHE / "map_rules.json").read_text())
        return {x["image_name"]: rule(maps[x["image_name"]], is_full(x), R[f"{method}/{is_full(x)}"]["q"],
                                      R[f"{method}/{is_full(x)}"]["smooth"]) for x in items}
    kind, body = spec.split(":", 1)
    thr = 0.5
    if "@" in body:
        body, t = body.rsplit("@", 1); thr = float(t)
    models = parse_models(body)
    probs = None
    for ckpt, feats in models:
        p = unet_probs(ckpt, feats, split, items, method)
        probs = p if probs is None else {k: probs[k] + p[k] for k in probs}
    return {k: (v / len(models) > thr).astype(np.uint8) for k, v in probs.items()}


def load_rects(method, split):
    r = {}
    for d in (CACHE, OUT):
        for f in sorted(d.glob(f"rect_{method}_{split}*.json")):
            r.update(json.loads(f.read_text()))
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--rect", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    for m in METHODS:
        ap.add_argument(f"--{m}", default="regime")
    a = ap.parse_args()
    train_pred, test_pred, n_rect = {}, {}, {}
    for m in METHODS:
        spec = getattr(a, m)
        te = [x for x in meta("test") if x["method"] == m]
        test_pred.update(base(spec, m, "test", te))
        if not a.skip_train:
            train_pred.update(base(spec, m, "train", [x for x in meta("train") if x["method"] == m]))
        n_rect[m] = 0
        if a.rect:
            rects = load_rects(m, "test")
            for x in te:
                r = rects.get(x["image_name"])
                if r is not None and not is_full(x) and r["acc"] >= RECT_THR[m]:
                    test_pred[x["image_name"]] = box_mask(tuple(r["box"])); n_rect[m] += 1
    if train_pred:
        report(train_pred, f"train base (optimistic for models trained on train) {vars(a)}")
    names = [x["image_name"] for x in meta("test")]
    masks = torch.from_numpy(np.stack([test_pred[n] for n in names]).astype(bool))
    assert masks.shape == (1020, 128, 128) and len(set(names)) == 1020
    sub = OUT / "submissions"; sub.mkdir(exist_ok=True)
    out = sub / a.out
    torch.save({"image_names": names, "masks": masks}, out)
    chk = torch.load(out, weights_only=True)
    assert chk["masks"].shape == (1020, 128, 128) and chk["image_names"] == names
    print("wrote", out, chk["masks"].dtype, f"{out.stat().st_size / 1e6:.1f} MB", "rect overrides:", n_rect,
          "test frac1:", {m: round(float(np.mean([test_pred[x['image_name']].mean() for x in meta('test') if x['method'] == m])), 3) for m in METHODS})
    try:
        dst = Path(__file__).parent / "submissions" / a.out; shutil.copy(out, dst); print("copied to", dst)
    except OSError as e:
        print("WARNING: copy to ours/submissions failed:", e)


if __name__ == "__main__":
    main()
