"""Build v11 from ~/wm_out/v11_choice.json (eval_v11.py). Same structure as final_v10.py plus:
- map families may include the local-message-decoder heads (lmdfx*/lmdall),
- in the PARTIAL regime, optional SAM refinement (mode chosen on OOF with the chosen decision) using the test SAM masks.
Rectangle overrides (PixelSeal, TrustMark) are applied last, as before."""
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from common import load_image, load_masks128, meta
from decision import apply_iso, clean_full_or_partial, emiou_mask, fit_iso
from eval_v10 import guide_of, probe_oof, unet_oof
from eval_v11 import MODES, sam_mask
from guided import guided
from map_rules import is_full
from predict import RECT_THR, load_rects
from rect_search import box_mask
from sam_masks import unpack
from wmloc.utils import miou_image

TTA = Path.home() / "wm_out" / "tta"; OUT = Path.home() / "wm_out"; SUB = OUT / "submissions"; gt = load_masks128()
FAMS = {"v10:fx+dil": ["fx", "dfx"], "lmd": ["lmdfx"], "fx+dil+lmd": ["fx", "dfx", "lmdfx"]}
ALLNAME = {"fx": "all", "dfx": "dall", "lmdfx": "lmdall"}


def ld_test(f):
    z = np.load(TTA / f); return {k[:-4]: z[k].astype(np.float32) for k in z.files if k.endswith("|tta")}


def units(m, variant):
    if m == "stegastamp":
        O = unet_oof(m, "_s2"); tests = [ld_test(f"unet_{m}_{m}_f{k}_s2_test.npz") for k in ["0", "1", "2", "3", "4", "-1"]]
        return O, tests
    fl = FAMS[variant]; pre = "" if m == "pixelseal" else "tm_"
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


def predec(q, cfg, guide):
    g, s, _ = cfg
    if g:
        q = guided(guide, q, g[0], g[1])
    if s:
        q = ndimage.gaussian_filter(q, s)
    return q


def main():
    choice = json.loads((OUT / "v11_choice.json").read_text())
    v10 = torch.load(SUB / "v10.pt", weights_only=True); names = v10["image_names"]; idx = {n: i for i, n in enumerate(names)}
    masks = v10["masks"].clone(); rep = {}
    for m in ("pixelseal", "trustmark", "stegastamp"):
        ch = choice[m]; O, tests = units(m, ch["maps"]); use_gf = m != "stegastamp"
        tr = [x for x in meta("train") if x["method"] == m]
        gfs = [None] + ([(r, e) for r in (2, 4, 8) for e in (1e-3, 1e-2)] if use_gf else [])
        cal, cfg, mode = {}, {}, {}
        Ztr = np.load(OUT / "sam" / f"masks_{m}_train.npz") if ch["sam"] else None
        for full in (True, False):
            grp = [x["image_name"] for x in tr if is_full(x) == full]
            cal[full] = fit_iso(np.concatenate([O[n].ravel() for n in grp]), np.concatenate([gt[n].ravel() for n in grp]).astype(float))
            Q = {n: apply_iso(cal[full], O[n]) for n in grp}; Gd = {n: guide_of(n) for n in grp}
            sc = {}
            for g in gfs:
                for s in (0, 1):
                    for c in (0, 16):
                        sc[(g, s, c)] = np.mean([miou_image(clean_full_or_partial(emiou_mask(predec(Q[n], (g, s, c), Gd[n])), c), gt[n]) for n in grp])
            cfg[full] = max(sc, key=sc.get); mode[full] = ("none",)
            if ch["sam"] and not full:
                ms = {}
                for md in MODES:
                    v = []
                    for n in grp:
                        qd = predec(Q[n], cfg[full], Gd[n]); b = clean_full_or_partial(emiou_mask(qd), cfg[full][2])
                        segs = unpack(Ztr[n]) if n in Ztr.files else np.zeros((0, 128, 128), np.uint8)
                        v.append(miou_image(sam_mask(md, qd, b, segs, cfg[full][2]), gt[n]))
                    ms[md] = np.mean(v)
                mode[full] = max(ms, key=ms.get)
            rep[f"{m}/{full}"] = {"maps": ch["maps"], "cfg": str(cfg[full]), "sam_mode": str(mode[full])}
        Zte = np.load(OUT / "sam" / f"masks_{m}_test.npz") if ch["sam"] else None
        rects = load_rects(m, "test") if m != "stegastamp" else {}
        fr = {True: [], False: []}; nrect = 0; changed = 0
        for x in [x for x in meta("test") if x["method"] == m]:
            n = x["image_name"]; full = is_full(x)
            q = np.mean([apply_iso(cal[full], t[n]) for t in tests], 0)
            guide = load_image("test", n).astype(np.float32).mean(2).reshape(128, 4, 128, 4).mean((1, 3)) / 255
            qd = predec(q, cfg[full], guide); mk = clean_full_or_partial(emiou_mask(qd), cfg[full][2])
            if mode[full] != ("none",):
                segs = unpack(Zte[n]) if n in Zte.files else np.zeros((0, 128, 128), np.uint8)
                mk = sam_mask(mode[full], qd, mk, segs, cfg[full][2])
            fr[full].append(mk.mean())
            r = rects.get(n)
            if r is not None and not full and r["acc"] >= RECT_THR[m]:
                mk = box_mask(tuple(r["box"])); nrect += 1
            new = torch.from_numpy(mk.astype(bool)); changed += float((new != masks[idx[n]]).float().mean()); masks[idx[n]] = new
        for full in (True, False):
            rep[f"{m}/{full}"]["test_frac"] = round(float(np.mean(fr[full])), 3)
        print(f"{m}: {ch} | rect overrides {nrect} | cells changed vs v10 {changed / 340:.3f}", flush=True)
    for k, v in rep.items():
        print(f"   {k:16s} {v}")
    assert masks.shape == (1020, 128, 128) and len(set(names)) == 1020
    out = SUB / "v11.pt"; torch.save({"image_names": names, "masks": masks}, out)
    chk = torch.load(out, weights_only=True); assert chk["image_names"] == names and chk["masks"].dtype == torch.bool
    (OUT / "v11_config.json").write_text(json.dumps(rep, indent=1))
    print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB")
    try:
        import shutil; shutil.copy(out, Path(__file__).parent / "submissions" / "v11.pt"); print("copied to ours/submissions/v11.pt")
    except OSError as e:
        print("WARNING copy failed:", e)


if __name__ == "__main__":
    main()
