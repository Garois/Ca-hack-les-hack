"""Integrate the PixelSeal probe into the submission (v7 = v6 with PixelSeal masks replaced).
1. OOF probabilities of the fixed-epoch fold heads (heldprobs_fx0..4) -> per predicted regime, choose threshold and
   small-component cleanup on OOF; report per sub-type vs the current pipeline, with the rectangle override where a
   search result exists (acc >= RECT_THR).
2. Test: mean probability of the fold heads + the all-train head, same rule, rectangle override.
    run.sh ps_final.py --base v6.pt --out v7.pt"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import CACHE, load_masks128, meta
from global_rect import OUT as RECT_OUT
from map_rules import is_full, rule as cur_rule
from oof import postproc
from predict import RECT_THR, load_rects
from ps_probe import OUT as PROBE, subtype
from rect_search import box_mask
from wmloc.utils import miou_image

SUB = Path.home() / "wm_out" / "submissions"


def load_probs(pattern):
    d, n = {}, 0
    for f in sorted(PROBE.glob(pattern)):
        z = np.load(f); n += 1
        for k in z.files:
            d.setdefault(k, []).append(z[k].astype(np.float32))
    return {k: np.mean(v, 0) for k, v in d.items()}, n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default="v6.pt"); ap.add_argument("--out", default="v7.pt")
    ap.add_argument("--held", default="heldprobs_fx*.npz"); ap.add_argument("--test", default="testprobs_fx*.npz,testprobs_all.npz")
    ap.add_argument("--method", default="pixelseal")
    ap.add_argument("--blends", default="1.0,0.75,0.5,0.25,0.0",
                    help="probe weights tried when --unet is given (v8 was built with 1.0,0.75,0.5)")
    ap.add_argument("--unet", default="", help="comma list of U-Net ckpts to blend with (OOF from their *_valprobs)")
    a = ap.parse_args()
    M = a.method
    gt = load_masks128(); R = json.loads((CACHE / "map_rules.json").read_text())
    S4 = dict(np.load(CACHE / f"maps_{M}.npz")) if (CACHE / f"maps_{M}.npz").exists() else {}
    cur_cfg = json.loads((CACHE / "final_config.json").read_text())
    oof, nh = load_probs(a.held)
    uoof = {}
    if a.unet:
        from oof import valprobs
        for f in range(5):
            uoof.update(valprobs(f"{M}_f{f}_s2"))
    tr = [x for x in meta("train") if x["method"] == M and x["image_name"] in oof]
    rects_tr = {}
    for f in list(CACHE.glob(f"rect_{M}_train*.json")) + list(RECT_OUT.glob(f"rect_{M}_val0*.json")):
        rects_tr.update(json.loads(f.read_text()))
    print(f"[{M}] OOF probabilities for {len(tr)}/170 train images from {nh} fold files; U-Net OOF for {len(uoof)}")
    blends = tuple(float(b) for b in a.blends.split(",")) if uoof else (1.0,)
    cfg = {}
    for full in (True, False):
        grp = [x for x in tr if is_full(x) == full]
        P = lambda n, b: (b * oof[n] + (1 - b) * uoof[n]) if b < 1 else oof[n]
        cands = {(t, mc, b): np.mean([miou_image(postproc(P(x["image_name"], b), full, t, mc), gt[x["image_name"]]) for x in grp])
                 for t in (0.3, 0.4, 0.5, 0.6, 0.7) for mc in (0, 4, 16) for b in blends}
        best = max(cands, key=cands.get); cfg[full] = best
        print(f"pred_full={full!s:5s} n={len(grp)} best thr={best[0]} min={best[1]} blend={best[2]} OOF={cands[best]:.4f} "
              f"(probe only thr0.5: {cands[(0.5, 0, 1.0)]:.4f})")
    rows = {}
    for x in tr:
        n = x["image_name"]; full = is_full(x); g = gt[n]; t, mc, b = cfg[full]
        new = postproc(b * oof[n] + (1 - b) * uoof[n] if b < 1 else oof[n], full, t, mc)
        ch = cur_cfg[f"{M}/{full}"]["choice"]
        if ch == "maprule":
            r = R[f"{M}/{full}"]; cur = cur_rule(S4[n], full, r["q"], r["smooth"])
        elif ch == "const":
            cur = np.full((128, 128), int(full), np.uint8)
        else:
            tt, mm = ch[5:].split("/min"); cur = postproc(uoof[n], full, float(tt), int(mm)) if n in uoof else cur_rule(S4[n], full, R[f"{M}/{full}"]["q"], R[f"{M}/{full}"]["smooth"])
        rr = rects_tr.get(n)
        if rr is not None and not full and rr["acc"] >= RECT_THR[M]:
            new = cur = box_mask(tuple(rr["box"]))
        rows.setdefault(subtype(x, g), []).append((miou_image(cur, g), miou_image(new, g), rr is not None))
    tot_c, tot_n, N = 0, 0, sum(len(v) for v in rows.values())
    for k, v in sorted(rows.items()):
        v = np.array(v, dtype=float); tot_c += v[:, 0].sum(); tot_n += v[:, 1].sum()
        print(f"  {k:15s} n={len(v):3d} current={v[:, 0].mean():.4f} -> probe={v[:, 1].mean():.4f}  (rect searched on {int(v[:, 2].sum())})")
    print(f"[{M}] train OOF: current {tot_c / N:.4f} -> probe {tot_n / N:.4f}  (rect override only where searched: "
          f"partial/rect images without a search keep their base mask, so both numbers are pessimistic for that sub-type)")
    # test
    tp, nt = {}, 0
    for pat in a.test.split(","):
        d, n = load_probs(pat); nt += n
        for k, v in d.items():
            tp.setdefault(k, []).append((v, n))
    tp = {k: sum(v * n for v, n in lst) / sum(n for _, n in lst) for k, lst in tp.items()}
    te = [x for x in meta("test") if x["method"] == M]
    assert all(x["image_name"] in tp for x in te), "missing test probabilities"
    if any(cfg[f][2] < 1 for f in cfg):
        from predict import unet_probs
        models = [c.strip() for c in a.unet.split(",")]
        up = None
        for c in models:
            p = unet_probs(c, "feats2", "test", te, M); up = p if up is None else {k: up[k] + p[k] for k in up}
        up = {k: v / len(models) for k, v in up.items()}
    rects = load_rects(M, "test")
    sub = torch.load(SUB / a.base, weights_only=True); names = sub["image_names"]; masks = sub["masks"].clone()
    idx = {n: i for i, n in enumerate(names)}; nrect = 0; changed = 0
    for x in te:
        n = x["image_name"]; full = is_full(x); t, mc, b = cfg[full]
        mk = postproc(b * tp[n] + (1 - b) * up[n] if b < 1 else tp[n], full, t, mc)
        r = rects.get(n)
        if r is not None and not full and r["acc"] >= RECT_THR[M]:
            mk = box_mask(tuple(r["box"])); nrect += 1
        new = torch.from_numpy(mk.astype(bool)); changed += int((new != masks[idx[n]]).any()); masks[idx[n]] = new
    assert masks.shape == (1020, 128, 128) and len(set(names)) == 1020
    out = SUB / a.out; torch.save({"image_names": names, "masks": masks}, out)
    chk = torch.load(out, weights_only=True); assert chk["image_names"] == names
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): {nt} test prob files averaged, {nrect} rect overrides, "
          f"{changed}/340 {M} masks changed vs {a.base}; {M} frac1 {masks[[idx[x['image_name']] for x in te]].float().mean():.3f}")
    try:
        import shutil; shutil.copy(out, Path(__file__).parent / "submissions" / a.out); print("copied to ours/submissions/")
    except OSError as e:
        print("WARNING copy failed:", e)


if __name__ == "__main__":
    main()
