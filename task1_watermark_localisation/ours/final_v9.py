"""Build v9 (PLAN_V9): for every method, 4-shift TTA probabilities -> per-model isotonic calibration (fitted on ALL OOF
TTA maps of the same family, per predicted regime) -> mean over the 6 models -> expected-mIoU decoding (smoothing s and
cleanup c chosen on OOF per regime) -> exact-rectangle override (PixelSeal, TrustMark: acc >= 0.8).
TrustMark family = 0.75 probe + 0.25 U-Net, blended model by model (fold k probe with fold k U-Net, all with all).
Sanity: predicted watermark fraction per regime, OOF vs test; cells changed vs v8."""
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from common import load_masks128, meta
from decision import apply_iso, clean_full_or_partial, emiou_mask, fit_iso
from map_rules import is_full
from predict import RECT_THR, load_rects
from rect_search import box_mask
from tta_eval import maps as oof_maps
from wmloc.utils import miou_image

TTA = Path.home() / "wm_out" / "tta"; SUB = Path.home() / "wm_out" / "submissions"; gt = load_masks128()


def test_models(m):
    """List of per-model TTA test maps (dict name -> 128x128) for the method's family."""
    def ld(f):
        z = np.load(TTA / f); return {k[:-4]: z[k].astype(np.float32) for k in z.files if k.endswith("|tta")}
    if m == "stegastamp":
        return [ld(f"unet_{m}_{m}_f{k}_s2_test.npz") for k in ["0", "1", "2", "3", "4", "-1"]]
    pre = "" if m == "pixelseal" else "tm_"
    probes = [ld(f"probe_{m}_{pre}fx{k}_test.npz") for k in range(5)] + [ld(f"probe_{m}_{pre}all_test.npz")]
    if m == "pixelseal":
        return probes
    unets = [ld(f"unet_{m}_{m}_f{k}_s2_test.npz") for k in ["0", "1", "2", "3", "4", "-1"]]
    return [{n: 0.75 * p[n] + 0.25 * u[n] for n in p} for p, u in zip(probes, unets)]


def decode(q, s, c):
    return clean_full_or_partial(emiou_mask(ndimage.gaussian_filter(q, s) if s else q), c)


def main():
    base = torch.load(SUB / "v8.pt", weights_only=True); names = base["image_names"]; idx = {n: i for i, n in enumerate(names)}
    masks = base["masks"].clone(); report = {}
    for m in ("pixelseal", "trustmark", "stegastamp"):
        tr = [x for x in meta("train") if x["method"] == m]; P = oof_maps(m, True)
        cal, cfg = {}, {}
        for full in (True, False):
            grp = [x["image_name"] for x in tr if is_full(x) == full]
            cal[full] = fit_iso(np.concatenate([P[n].ravel() for n in grp]), np.concatenate([gt[n].ravel() for n in grp]).astype(float))
            Q = {n: apply_iso(cal[full], P[n]) for n in grp}
            sc = {(s, c): np.mean([miou_image(decode(Q[n], s, c), gt[n]) for n in grp]) for s in (0, 1) for c in (0, 16)}
            cfg[full] = max(sc, key=sc.get)
            report[f"{m}/{full}"] = {"s": cfg[full][0], "c": cfg[full][1], "oof_in_sample": round(float(sc[cfg[full]]), 4),
                                     "oof_frac": round(float(np.mean([decode(Q[n], *cfg[full]).mean() for n in grp])), 3)}
        te = [x for x in meta("test") if x["method"] == m]; TM = test_models(m); rects = load_rects(m, "test")
        fr = {True: [], False: []}; nrect = 0; changed = 0
        for x in te:
            n = x["image_name"]; full = is_full(x)
            q = np.mean([apply_iso(cal[full], mod[n]) for mod in TM], 0)
            mk = decode(q, *cfg[full]); fr[full].append(mk.mean())
            r = rects.get(n)
            if m in ("pixelseal", "trustmark") and r is not None and not full and r["acc"] >= RECT_THR[m]:
                mk = box_mask(tuple(r["box"])); nrect += 1
            new = torch.from_numpy(mk.astype(bool)); changed += float((new != masks[idx[n]]).float().mean()); masks[idx[n]] = new
        for full in (True, False):
            report[f"{m}/{full}"]["test_frac"] = round(float(np.mean(fr[full])), 3)
        print(f"{m}: models {len(TM)}, rect overrides {nrect}, mean cells changed vs v8 {changed / len(te):.3f}")
    for k, v in report.items():
        print(f"   {k:16s} {v}")
    assert masks.shape == (1020, 128, 128) and len(set(names)) == 1020
    out = SUB / "v9.pt"; torch.save({"image_names": names, "masks": masks}, out)
    chk = torch.load(out, weights_only=True); assert chk["image_names"] == names and chk["masks"].dtype == torch.bool
    (Path.home() / "wm_out" / "v9_config.json").write_text(json.dumps(report, indent=1))
    print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB")
    try:
        import shutil; shutil.copy(out, Path(__file__).parent / "submissions" / "v9.pt"); print("copied to ours/submissions/v9.pt")
    except OSError as e:
        print("WARNING copy failed:", e)


if __name__ == "__main__":
    main()
