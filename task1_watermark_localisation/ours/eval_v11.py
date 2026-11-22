"""PLAN_V11 evaluation (run inside the Slurm job). Nested CV everywhere.
Stage 1 (PixelSeal, TrustMark): map variants v10 'fx+dil' vs 'lmd' vs 'fx+dil+lmd' (v10 decision with guided filter).
Stage 2 (all methods, PARTIAL regime only): SAM segment refinement on top of the per-fold v10 decision.
  modes: none | vote(tau) | greedy (expected-mIoU, parameter-free) | segavg (smallest-segment mean, parameter-free)
         | mix(alpha) = alpha*q + (1-alpha)*segavg
  The v10 decision (g, s, c) is chosen per fold without SAM, then the SAM mode is chosen per fold (two-stage nested).
Writes ~/wm_out/v11_choice.json."""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from common import load_masks128
from decision import apply_iso, boot, clean_full_or_partial, emiou_mask, fit_iso
from eval_v10 import evaluate as eval_maps, guide_of, probe_oof, unet_oof
from guided import guided
from map_rules import is_full
from ps_probe import subtype
from rect_search import box_mask
from sam_masks import unpack
from train_seg import folds
from wmloc.utils import miou_image

OUT = Path.home() / "wm_out"; gt = load_masks128(); N = 128 * 128


def family_maps(m, fams):
    pr = [probe_oof(m, f) for f in fams]; P = {n: np.mean([p[n] for p in pr], 0) for n in pr[0]}
    if m == "trustmark":
        U = unet_oof(m, "_s2"); P = {n: 0.75 * P[n] + 0.25 * U[n] for n in P}
    return P


def e_miou(mask, q, T):
    S = float(q[mask == 1].sum()); K = float(mask.sum())
    e1 = S / (K + T - S) if K + T - S > 1e-9 else 1.0
    e0 = (N - K - T + S) / (N - S) if N - S > 1e-9 else 1.0
    return 0.5 * (e1 + e0)


def seg_avg(q, segs):
    if not len(segs):
        return q
    lab = np.full(q.shape, -1); order = np.argsort([-s.sum() for s in segs])       # large first, small overwrite
    for i in order:
        lab[segs[i] == 1] = i
    out = q.copy()
    for i in np.unique(lab[lab >= 0]):
        out[lab == i] = q[lab == i].mean()
    return out


def sam_mask(mode, qd, base, segs, c):
    if mode == ("none",) or not len(segs):
        return base
    if mode[0] == "vote":
        sel = [s for s in segs if qd[s == 1].mean() >= mode[1]]
        return np.clip(np.sum(sel, 0), 0, 1).astype(np.uint8) if sel else base
    if mode[0] == "greedy":
        T = float(qd.sum()); cur = np.zeros_like(base); best = e_miou(cur, qd, T)
        for i in np.argsort([-qd[s == 1].mean() for s in segs]):
            cand = np.maximum(cur, segs[i]); v = e_miou(cand, qd, T)
            if v > best:
                cur, best = cand, v
        return cur if cur.any() else base
    if mode[0] == "segavg":
        return clean_full_or_partial(emiou_mask(seg_avg(qd, segs)), c)
    if mode[0] == "mix":
        return clean_full_or_partial(emiou_mask(mode[1] * qd + (1 - mode[1]) * seg_avg(qd, segs)), c)
    raise ValueError(mode)


MODES = [("none",)] + [("vote", t) for t in (0.3, 0.4, 0.5, 0.6)] + [("greedy",), ("segavg",)] + [("mix", a) for a in (0.25, 0.5, 0.75)]


def stage2(m, P, items, fold_of, G, use_gf, SAM, rects=None):
    names = [x["image_name"] for x in items]; X = {x["image_name"]: x for x in items}; fullp = {n: is_full(X[n]) for n in names}
    gfs = [None] + ([(r, e) for r in (2, 4, 8) for e in (1e-3, 1e-2)] if use_gf else []); cfgs = [(g, s, c) for g in gfs for s in (0, 1) for c in (0, 16)]
    Q = {}
    for f in range(5):
        for full in (True, False):
            tr = [n for n in names if fold_of[n] != f and fullp[n] == full]
            cal = fit_iso(np.concatenate([P[n].ravel() for n in tr]), np.concatenate([gt[n].ravel() for n in tr]).astype(float))
            for n in names:
                if fold_of[n] == f and fullp[n] == full:
                    Q[n] = apply_iso(cal, P[n])
    memo_q, memo_b = {}, {}

    def qd(n, cfg):
        if (n, cfg) not in memo_q:
            g, s, _ = cfg; q = Q[n]
            if g:
                q = guided(G[n], q, g[0], g[1])
            if s:
                q = ndimage.gaussian_filter(q, s)
            memo_q[(n, cfg)] = q
        return memo_q[(n, cfg)]

    def base(n, cfg):
        if (n, cfg) not in memo_b:
            memo_b[(n, cfg)] = clean_full_or_partial(emiou_mask(qd(n, cfg)), cfg[2])
        return memo_b[(n, cfg)]

    segs = {n: unpack(SAM[n]) if n in SAM else np.zeros((0, 128, 128), np.uint8) for n in names}
    memo_s = {}

    def score(n, cfg, mode):
        if (n, cfg, mode) not in memo_s:
            mk = base(n, cfg) if (fullp[n] or mode == ("none",)) else sam_mask(mode, qd(n, cfg), base(n, cfg), segs[n], cfg[2])
            r = rects.get(n) if rects is not None else None
            if r is not None and not fullp[n] and r["acc"] >= 0.8:             # rectangle override last, as in test
                mk = box_mask(tuple(r["box"]))
            memo_s[(n, cfg, mode)] = miou_image(mk, gt[n])
        return memo_s[(n, cfg, mode)]

    chosen = {}; forced = {md: {} for md in MODES}; auto = {}
    for f in range(5):
        for full in (True, False):
            tr = [n for n in names if fold_of[n] != f and fullp[n] == full]; te = [n for n in names if fold_of[n] == f and fullp[n] == full]
            cfg = max(cfgs, key=lambda c: np.mean([score(n, c, ("none",)) for n in tr]))
            mode = ("none",) if full else max(MODES, key=lambda md: np.mean([score(n, cfg, md) for n in tr]))
            chosen[(f, full)] = (cfg, mode)
            for n in te:
                auto[n] = score(n, cfg, mode)
                for md in MODES:
                    forced[md][n] = score(n, cfg, md)
    return names, X, forced, auto, chosen


def report(tag, r, ref, names, X):
    d = [r[n] - ref[n] for n in names]; mu, lo, hi = boot(d); by = {}
    for n in names:
        by.setdefault(subtype(X[n], gt[n]), []).append(r[n] - ref[n])
    print(f"   {tag:26s} {np.mean([r[n] for n in names]):.4f} | vs ref {mu:+.4f} (95% CI {lo:+.4f}..{hi:+.4f}) | "
          + " ".join(f"{k}:{np.mean(v):+.3f}" for k, v in sorted(by.items())), flush=True)
    return lo, float(np.mean([r[n] for n in names]))


def main():
    choice = {}
    for m in ("pixelseal", "trustmark", "stegastamp"):
        items, fold_of = folds(m); names = [x["image_name"] for x in items]; X = {x["image_name"]: x for x in items}
        G = {n: guide_of(n) for n in names}
        if m == "stegastamp":
            maps = {"v10:unet": unet_oof(m, "_s2")}; use_gf = False; best_map = "v10:unet"
        else:
            use_gf = True
            maps = {"v10:fx+dil": family_maps(m, ["fx", "dfx"])}
            lm = probe_oof(m, "lmdfx")
            if len(lm) == 170:
                maps["lmd"] = family_maps(m, ["lmdfx"]); maps["fx+dil+lmd"] = family_maps(m, ["fx", "dfx", "lmdfx"])
            print(f"== {m} stage 1 (maps, v10 decision)", flush=True)
            ref = eval_maps(m, maps["v10:fx+dil"], items, fold_of, G, use_gf=True); best_map, best_mean = "v10:fx+dil", np.mean(list(ref.values()))
            for k in [k for k in maps if k != "v10:fx+dil"]:
                r = eval_maps(m, maps[k], items, fold_of, G, use_gf=True); lo, mean = report(k, r, ref, names, X)
                if lo > 0 and mean > best_mean:
                    best_map, best_mean = k, mean
        print(f"== {m} stage 2 (SAM, partial regime) on maps '{best_map}'", flush=True)
        SAM = {}
        f = OUT / "sam" / f"masks_{m}_train.npz"
        if f.exists():
            z = np.load(f); SAM = {k: z[k] for k in z.files}
        names, X, forced, auto, chosen = stage2(m, maps[best_map], items, fold_of, G, use_gf, SAM)
        ref2 = forced[("none",)]
        for md in MODES[1:]:
            report(f"SAM forced {md}", forced[md], ref2, names, X)
        lo, mean = report("SAM auto (nested choice)", auto, ref2, names, X)
        print("   chosen partial modes per fold:", [str(chosen[(f, False)][1]) for f in range(5)], flush=True)
        choice[m] = {"maps": best_map, "sam": bool(lo > 0)}
        print(f"   -> {choice[m]}", flush=True)
    (OUT / "v11_choice.json").write_text(json.dumps(choice, indent=1))


if __name__ == "__main__":
    main()
