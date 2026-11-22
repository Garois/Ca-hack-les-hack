"""Build v10 from ~/wm_out/v10_choice.json (written by eval_v10.py).
Per method: unit_k = mean over the chosen families of the fold-k model maps (TrustMark: 0.75 probe + 0.25 U-Net),
'all' unit likewise; isotonic calibration per regime fitted on the OOF units; at test each unit is calibrated, the
6 units are averaged, then optional guided filter (guide = test image luminance on the 128 grid), Gaussian smoothing
s, expected-mIoU decoding, cleanup c ((g, s, c) chosen per regime on OOF), rectangle overrides (PixelSeal, TrustMark
as before; StegaStamp seeded search if adopted). Sanity: predicted fractions OOF vs test, cells changed vs v9."""
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from common import load_image, load_masks128, meta
from decision import apply_iso, clean_full_or_partial, emiou_mask, fit_iso
from eval_v10 import guide_of, probe_oof, unet_oof
from guided import guided
from map_rules import is_full
from predict import RECT_THR, load_rects
from rect_search import box_mask
from wmloc.utils import miou_image

TTA = Path.home() / "wm_out" / "tta"; OUT = Path.home() / "wm_out"; SUB = OUT / "submissions"; gt = load_masks128()
FAMS = {"v9:fx": ["fx"], "dil": ["dfx"], "seeds3": ["fx", "s1fx", "s2fx"], "fx+dil": ["fx", "dfx"],
        "all4": ["fx", "s1fx", "s2fx", "dfx"], "v9:unet": ["_s2"], "unet+seed": ["_s2", "_s2b"]}
ALLNAME = {"fx": "all", "dfx": "dall", "s1fx": "s1all", "s2fx": "s2all"}


def ld_test(f):
    z = np.load(TTA / f); return {k[:-4]: z[k].astype(np.float32) for k in z.files if k.endswith("|tta")}


def units(m, variant):
    """(oof_map dict, list of 6 test unit dicts)."""
    fl = FAMS[variant]
    if m == "stegastamp":
        oof = [unet_oof(m, t) for t in fl]; O = {n: np.mean([o[n] for o in oof], 0) for n in oof[0]}
        tests = []
        for k in ["0", "1", "2", "3", "4", "-1"]:
            ts = [ld_test(f"unet_{m}_{m}_f{k}{t}_test.npz") for t in fl]; tests.append({n: np.mean([t[n] for t in ts], 0) for n in ts[0]})
        return O, tests
    pre = "" if m == "pixelseal" else "tm_"
    oof = [probe_oof(m, f) for f in fl]; O = {n: np.mean([o[n] for o in oof], 0) for n in oof[0]}
    tests = []
    for k in range(6):
        ts = [ld_test(f"probe_{m}_{pre}{f}{k}_test.npz" if k < 5 else f"probe_{m}_{pre}{ALLNAME[f]}_test.npz") for f in fl]
        tests.append({n: np.mean([t[n] for t in ts], 0) for n in ts[0]})
    if m == "trustmark":
        U = unet_oof(m, "_s2"); O = {n: 0.75 * O[n] + 0.25 * U[n] for n in O}
        for k, kk in enumerate(["0", "1", "2", "3", "4", "-1"]):
            u = ld_test(f"unet_{m}_{m}_f{kk}_s2_test.npz"); tests[k] = {n: 0.75 * tests[k][n] + 0.25 * u[n] for n in tests[k]}
    return O, tests


def decode(q, cfg, guide):
    g, s, c = cfg
    if g:
        q = guided(guide, q, g[0], g[1])
    if s:
        q = ndimage.gaussian_filter(q, s)
    return clean_full_or_partial(emiou_mask(q), c)


def main():
    choice = json.loads((OUT / "v10_choice.json").read_text())
    v9 = torch.load(SUB / "v9.pt", weights_only=True); names = v9["image_names"]; idx = {n: i for i, n in enumerate(names)}
    masks = v9["masks"].clone(); rep = {}
    for m in ("pixelseal", "trustmark", "stegastamp"):
        ch = choice[m]; O, tests = units(m, ch["variant"])
        tr = [x for x in meta("train") if x["method"] == m]
        gfs = [None] + ([(r, e) for r in (2, 4, 8) for e in (1e-3, 1e-2)] if ch["guided"] else [])
        cal, cfg = {}, {}
        for full in (True, False):
            grp = [x["image_name"] for x in tr if is_full(x) == full]
            cal[full] = fit_iso(np.concatenate([O[n].ravel() for n in grp]), np.concatenate([gt[n].ravel() for n in grp]).astype(float))
            Q = {n: apply_iso(cal[full], O[n]) for n in grp}; Gd = {n: guide_of(n) for n in grp}
            sc = {(g, s, c): np.mean([miou_image(decode(Q[n], (g, s, c), Gd[n]), gt[n]) for n in grp])
                  for g in gfs for s in (0, 1) for c in (0, 16)}
            cfg[full] = max(sc, key=sc.get)
            rep[f"{m}/{full}"] = {"variant": ch["variant"], "cfg": str(cfg[full]), "oof_in_sample": round(float(sc[cfg[full]]), 4),
                                  "oof_frac": round(float(np.mean([decode(Q[n], cfg[full], Gd[n]).mean() for n in grp])), 3)}
        rects = load_rects(m, "test") if m != "stegastamp" else (
            json.loads((OUT / "ssrect_test.json").read_text()) if choice.get("stegastamp_rect") else {})
        thr = RECT_THR[m] if m != "stegastamp" else 0.8
        fr = {True: [], False: []}; nrect = 0; changed = 0
        for x in [x for x in meta("test") if x["method"] == m]:
            n = x["image_name"]; full = is_full(x)
            q = np.mean([apply_iso(cal[full], t[n]) for t in tests], 0)
            guide = load_image("test", n).astype(np.float32).mean(2).reshape(128, 4, 128, 4).mean((1, 3)) / 255
            mk = decode(q, cfg[full], guide); fr[full].append(mk.mean())
            r = rects.get(n)
            if r is not None and not full and r["acc"] >= thr:
                mk = box_mask(tuple(r["box"])); nrect += 1
            new = torch.from_numpy(mk.astype(bool)); changed += float((new != masks[idx[n]]).float().mean()); masks[idx[n]] = new
        for full in (True, False):
            rep[f"{m}/{full}"]["test_frac"] = round(float(np.mean(fr[full])), 3)
        print(f"{m}: variant {ch['variant']} guided {ch['guided']} | rect overrides {nrect} | cells changed vs v9 {changed / 340:.3f}", flush=True)
    for k, v in rep.items():
        print(f"   {k:16s} {v}")
    assert masks.shape == (1020, 128, 128) and len(set(names)) == 1020
    out = SUB / "v10.pt"; torch.save({"image_names": names, "masks": masks}, out)
    chk = torch.load(out, weights_only=True); assert chk["image_names"] == names and chk["masks"].dtype == torch.bool
    (OUT / "v10_config.json").write_text(json.dumps(rep, indent=1))
    print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB")
    try:
        import shutil; shutil.copy(out, Path(__file__).parent / "submissions" / "v10.pt"); print("copied to ours/submissions/v10.pt")
    except OSError as e:
        print("WARNING copy failed:", e)


if __name__ == "__main__":
    main()
