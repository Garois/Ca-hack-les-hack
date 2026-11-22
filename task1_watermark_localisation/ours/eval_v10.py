"""PLAN_V10 B evaluation (CPU). Map variants (OOF, 4-shift TTA) x v10 decision (isotonic calibration per regime fitted on
the other folds -> optional guided filter -> optional Gaussian smoothing -> expected-mIoU decoding -> cleanup), all
selected per regime inside nested CV. A 'unit' = mean over the families/seeds of the fold-k models (the same unit is
calibrated then averaged over folds at test time). TrustMark units are blended 0.75 probe + 0.25 U-Net.
Also: StegaStamp seeded-rectangle override (acc >= 0.8) on predicted-partial images, false-positive check.
Writes ~/wm_out/v10_choice.json with the chosen variant per method."""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from common import load_masks128, load_image
from decision import apply_iso, boot, clean_full_or_partial, emiou_mask, fit_iso
from guided import guided
from map_rules import is_full
from ps_probe import subtype
from rect_search import box_mask
from train_seg import folds
from wmloc.utils import miou_image

TTA = Path.home() / "wm_out" / "tta"; OUT = Path.home() / "wm_out"; gt = load_masks128()


def load_oof(files):
    d = {}
    for f in files:
        if f.exists():
            z = np.load(f); d.update({k[:-4]: z[k].astype(np.float32) for k in z.files if k.endswith("|tta")})
    return d


def probe_oof(m, fam):
    pre = "" if m == "pixelseal" else "tm_"
    return load_oof([TTA / f"probe_{m}_{pre}{fam}{k}_oof.npz" for k in range(5)])


def unet_oof(m, tag):
    return load_oof([TTA / f"unet_{m}_{m}_f{k}{tag}_oof.npz" for k in range(5)])


def variants(m):
    V = {}
    if m == "stegastamp":
        a, b = unet_oof(m, "_s2"), unet_oof(m, "_s2b")
        V["v9:unet"] = a
        if len(b) == len(a):
            V["unet+seed"] = {n: (a[n] + b[n]) / 2 for n in a}
        return V
    fams = {f: probe_oof(m, f) for f in ("fx", "dfx", "s1fx", "s2fx")}
    fams = {f: v for f, v in fams.items() if len(v) == 170}
    combos = {"v9:fx": ["fx"], "dil": ["dfx"], "seeds3": ["fx", "s1fx", "s2fx"], "fx+dil": ["fx", "dfx"],
              "all4": ["fx", "s1fx", "s2fx", "dfx"]}
    U = unet_oof(m, "_s2") if m == "trustmark" else None
    for name, fl in combos.items():
        if all(f in fams for f in fl):
            pr = {n: np.mean([fams[f][n] for f in fl], 0) for n in fams["fx"]}
            V[name] = {n: 0.75 * pr[n] + 0.25 * U[n] for n in pr} if U is not None else pr
    return V


def guide_of(n):
    return load_image("train", n).astype(np.float32).mean(2).reshape(128, 4, 128, 4).mean((1, 3)) / 255


def evaluate(m, P, items, fold_of, G, use_gf, rects=None):
    names = [x["image_name"] for x in items]; X = {x["image_name"]: x for x in items}; fullp = {n: is_full(X[n]) for n in names}
    Q = {}
    for f in range(5):
        for full in (True, False):
            tr = [n for n in names if fold_of[n] != f and fullp[n] == full]
            cal = fit_iso(np.concatenate([P[n].ravel() for n in tr]), np.concatenate([gt[n].ravel() for n in tr]).astype(float))
            for n in names:
                if fold_of[n] == f and fullp[n] == full:
                    Q[n] = apply_iso(cal, P[n])
    gfs = [None] + ([(r, e) for r in (2, 4, 8) for e in (1e-3, 1e-2)] if use_gf else [])
    cfgs = [(g, s, c) for g in gfs for s in (0, 1) for c in (0, 16)]
    memo = {}

    def mask(n, cfg):
        if (n, cfg) not in memo:
            g, s, c = cfg; q = Q[n]
            if g:
                q = guided(G[n], q, g[0], g[1])
            if s:
                q = ndimage.gaussian_filter(q, s)
            memo[(n, cfg)] = clean_full_or_partial(emiou_mask(q), c)
        return memo[(n, cfg)]

    def sc(n, cfg):
        mk = mask(n, cfg)
        if rects is not None:
            r = rects.get(n)
            if r is not None and not fullp[n] and r["acc"] >= 0.8:
                mk = box_mask(tuple(r["box"]))
        return miou_image(mk, gt[n])

    out = {}
    for f in range(5):
        for full in (True, False):
            tr = [n for n in names if fold_of[n] != f and fullp[n] == full]
            te = [n for n in names if fold_of[n] == f and fullp[n] == full]
            best = max(cfgs, key=lambda c: np.mean([sc(n, c) for n in tr]))
            for n in te:
                out[n] = sc(n, best)
    return out


def report(tag, r, ref, names, X):
    d = [r[n] - ref[n] for n in names]; mu, lo, hi = boot(d); by = {}
    for n in names:
        by.setdefault(subtype(X[n], gt[n]), []).append(r[n] - ref[n])
    print(f"   {tag:28s} {np.mean([r[n] for n in names]):.4f} | vs v9 {mu:+.4f} (95% CI {lo:+.4f}..{hi:+.4f}) | "
          + " ".join(f"{k}:{np.mean(v):+.3f}" for k, v in sorted(by.items())), flush=True)
    return lo


def main():
    choice = {}
    for m in ("pixelseal", "trustmark", "stegastamp"):
        items, fold_of = folds(m); names = [x["image_name"] for x in items]; X = {x["image_name"]: x for x in items}
        G = {n: guide_of(n) for n in names}; V = variants(m)
        print(f"== {m}: variants {list(V)}", flush=True)
        v9key = [k for k in V if k.startswith("v9:")][0]
        ref = evaluate(m, V[v9key], items, fold_of, G, use_gf=False)            # v9 pipeline (no guided filter)
        best = (v9key, False, -1.0, ref)
        for k, P in V.items():
            for gf in ((False, True) if m != "stegastamp" else (False,)):
                if k == v9key and not gf:
                    continue
                r = evaluate(m, P, items, fold_of, G, use_gf=gf)
                lo = report(f"{k}{' +guided' if gf else ''}", r, ref, names, X)
                mean = np.mean(list(r.values()))
                if lo > 0 and mean > best[2]:
                    best = (k, gf, mean, r)
        if m == "stegastamp":
            fr = OUT / "ssrect_train.json"
            if fr.exists():
                rects = json.loads(fr.read_text()); P = V[best[0]]
                r = evaluate(m, P, items, fold_of, G, use_gf=False, rects=rects)
                lo = report(f"{best[0]} +SS rect override", r, ref, names, X)
                acc = {n: v["acc"] for n, v in rects.items()}
                fired = [n for n in acc if acc[n] >= 0.8]
                fp = [n for n in fired if subtype(X[n], gt[n]) != "partial/rect"]
                print(f"   SS rect override: fired on {len(fired)} train images, of which non-rectangles: {len(fp)} "
                      f"(rect recall {sum(1 for n in fired if subtype(X[n], gt[n]) == 'partial/rect')}/"
                      f"{sum(1 for n in names if subtype(X[n], gt[n]) == 'partial/rect')})")
                if lo > 0 and np.mean(list(r.values())) > best[2] - 1e-9:
                    choice[m + "_rect"] = True
        choice[m] = {"variant": best[0], "guided": best[1]}
        print(f"   -> chosen {choice[m]}", flush=True)
    (OUT / "v10_choice.json").write_text(json.dumps(choice, indent=1))


if __name__ == "__main__":
    main()
