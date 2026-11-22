"""W2 check: is the lower test watermark fraction in the full regime (TrustMark, StegaStamp) an artefact of averaging
6 calibrated models, or a property of the test images? Compares, on predicted-full test images: v8 fraction, v9 (mean of
calibrated models then decode), per-model decode (mean of fractions); and on train OOF: v8-rule fraction vs v9 decode."""
import numpy as np, torch
from pathlib import Path
from common import load_masks128, meta
from decision import apply_iso, fit_iso
from final_v9 import decode, test_models
from map_rules import is_full
from oof import postproc
from tta_eval import maps as oof_maps
gt = load_masks128(); SUB = Path.home() / "wm_out" / "submissions"
v8 = torch.load(SUB / "v8.pt", weights_only=True); idx = {n: i for i, n in enumerate(v8["image_names"])}
V8CFG = {"trustmark": (0.7, 0), "stegastamp": (0.7, 16)}
for m in ("trustmark", "stegastamp"):
    tr = [x for x in meta("train") if x["method"] == m and is_full(x)]; P = oof_maps(m, True); P0 = oof_maps(m, False)
    cal = fit_iso(np.concatenate([P[x["image_name"]].ravel() for x in tr]), np.concatenate([gt[x["image_name"]].ravel() for x in tr]).astype(float))
    oof_v8 = np.mean([postproc(P0[x["image_name"]], True, *V8CFG[m]).mean() for x in tr])
    oof_v9 = np.mean([decode(apply_iso(cal, P[x["image_name"]]), 1, 16).mean() for x in tr])
    gt_frac = np.mean([gt[x["image_name"]].mean() for x in tr])
    te = [x for x in meta("test") if x["method"] == m and is_full(x)]; TM = test_models(m)
    t_v8 = np.mean([v8["masks"][idx[x["image_name"]]].float().mean().item() for x in te])
    t_v9 = np.mean([decode(np.mean([apply_iso(cal, mod[x["image_name"]]) for mod in TM], 0), 1, 16).mean() for x in te])
    t_per = np.mean([np.mean([decode(apply_iso(cal, mod[x["image_name"]]), 1, 16).mean() for mod in TM]) for x in te])
    print(f"{m} full regime | TRAIN: GT {gt_frac:.3f}, v8-rule OOF {oof_v8:.3f}, v9-decode OOF {oof_v9:.3f} | "
          f"TEST: v8 {t_v8:.3f}, v9 (avg then decode) {t_v9:.3f}, per-model decode {t_per:.3f}")
