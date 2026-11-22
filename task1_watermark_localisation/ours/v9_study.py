"""Why was v9 (StegaStamp probe/U-Net blend) rejected? And is v8 itself robust?
1. Paired per-image differences (bootstrap CI) and NESTED cross-validation (parameters chosen on 4 folds, scored on
   the 5th) for: StegaStamp U-Net vs blend (v9), PixelSeal base vs probe (v7), TrustMark base vs probe-blend (v8).
2. Per-subtype AUC of probe vs U-Net (StegaStamp) and shape of the predicted masks on partial rectangles.
3. StegaStamp partial rectangles: (a) U-Net box accuracy vs GT, (b) decoder tolerance to moving ONE edge / translating.
CPU only (cached OOF probabilities)."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from common import CACHE, ROOT, bits, load_image, load_mask512, load_masks128, meta
from exp_ps_maps import auc
from map_rules import is_full, rule as cur_rule
from oof import postproc, valprobs
from ps_probe import subtype
from train_seg import folds
from wmloc.utils import miou_image

PROBE = Path.home() / "wm_out" / "ps_probe"
gt = load_masks128(); R = json.loads((CACHE / "map_rules.json").read_text())
GRID_T = (0.3, 0.4, 0.5, 0.6, 0.7); GRID_C = (0, 4, 16)


def load_probs(pattern):
    d = {}
    for f in sorted(PROBE.glob(pattern)):
        z = np.load(f); d.update({k: z[k].astype(np.float32) for k in z.files})
    return d


def nested(items, fold_of, score_fn, configs):
    """score_fn(x, cfg) -> mIoU. Per regime, choose cfg on the 4 other folds, score on the held fold."""
    out = {}
    for f in range(5):
        for full in (True, False):
            tr = [x for x in items if fold_of[x["image_name"]] != f and is_full(x) == full]
            te = [x for x in items if fold_of[x["image_name"]] == f and is_full(x) == full]
            best = max(configs, key=lambda c: np.mean([score_fn(x, c) for x in tr]))
            for x in te:
                out[x["image_name"]] = score_fn(x, best)
    return out


def boot_ci(d, n=2000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d)
    m = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return float(np.mean(d)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def study(method, probe_pat, base_kind):
    items, fold_of = folds(method)
    P = load_probs(probe_pat); U = {}
    for f in range(5):
        U.update(valprobs(f"{method}_f{f}_s2"))
    S4 = dict(np.load(CACHE / f"maps_{method}.npz")) if (CACHE / f"maps_{method}.npz").exists() else {}

    def base_score(x, c):                          # current base family
        n = x["image_name"]; full = is_full(x)
        if c == "maprule":
            r = R[f"{method}/{full}"]; return miou_image(cur_rule(S4[n], full, r["q"], r["smooth"]), gt[n])
        t, mc = c; return miou_image(postproc(U[n], full, t, mc), gt[n])

    def new_score(x, c):
        n = x["image_name"]; full = is_full(x); t, mc, b = c
        p = b * P[n] + (1 - b) * U[n] if b < 1 else P[n]
        return miou_image(postproc(p, full, t, mc), gt[n])

    base_cfgs = (["maprule"] if base_kind == "maprule" else []) + [(t, c) for t in GRID_T for c in GRID_C]
    new_cfgs = [(t, c, b) for t in GRID_T for c in GRID_C for b in (1.0, 0.75, 0.5, 0.25)]
    nb = nested(items, fold_of, base_score, base_cfgs); nn = nested(items, fold_of, new_score, new_cfgs)
    names = [x["image_name"] for x in items]
    d = [nn[n] - nb[n] for n in names]
    mean, lo, hi = boot_ci(d)
    print(f"[{method}] NESTED-CV (no selection bias): base {np.mean([nb[n] for n in names]):.4f} -> new "
          f"{np.mean([nn[n] for n in names]):.4f} | paired diff {mean:+.4f} (95% CI {lo:+.4f} .. {hi:+.4f})")
    by = {}
    for x in items:
        by.setdefault(subtype(x, gt[x["image_name"]]), []).append(nn[x["image_name"]] - nb[x["image_name"]])
    print("   per subtype diff: " + ", ".join(f"{k} {np.mean(v):+.3f} (n{len(v)})" for k, v in sorted(by.items())))
    return items, P, U


# ---------- 1. nested CV for v7 (PixelSeal), v8 (TrustMark), v9 (StegaStamp)
study("pixelseal", "heldprobs_fx*.npz", "maprule")
study("trustmark", "heldprobs_tm_fx*.npz", "unet")
items, P, U = study("stegastamp", "heldprobs_ss_fx*.npz", "unet")

# ---------- 2. StegaStamp: AUC per subtype, and what the probe does on rectangles
rows = {}
for x in items:
    n = x["image_name"]; g = gt[n]; k = subtype(x, g)
    rows.setdefault(k, []).append((auc(P[n].ravel(), g.ravel()), auc(U[n].ravel(), g.ravel()),
                                   np.corrcoef(P[n].ravel(), U[n].ravel())[0, 1]))
for k, v in sorted(rows.items()):
    v = np.array(v, dtype=float)
    print(f"[stegastamp] {k:15s} n={len(v):3d} AUC probe {np.nanmean(v[:, 0]):.3f} | U-Net {np.nanmean(v[:, 1]):.3f} | pixel corr {np.nanmean(v[:, 2]):.2f}")


def fill_of(m):
    lab, k = ndimage.label(m)
    if k == 0:
        return np.nan, 0.0
    big = lab == (1 + np.argmax(ndimage.sum(m, lab, range(1, k + 1))))
    ys, xs = np.where(big); return big.sum() / ((ys.ptp() + 1) * (xs.ptp() + 1)), float(m.mean())


rect = [x for x in items if subtype(x, gt[x["image_name"]]) == "partial/rect" and not is_full(x)]
fu, fp = [], []
for x in rect:
    n = x["image_name"]
    fu.append(fill_of(postproc(U[n], False, 0.3, 0))); fp.append(fill_of(postproc(0.5 * P[n] + 0.5 * U[n], False, 0.4, 16)))
fu, fp = np.array(fu, dtype=float), np.array(fp, dtype=float)
print(f"[stegastamp] partial/rect (n={len(rect)}): predicted-region rectangularity (bbox fill) U-Net {np.nanmedian(fu[:, 0]):.2f} vs blend "
      f"{np.nanmedian(fp[:, 0]):.2f}; predicted wm fraction U-Net {np.mean(fu[:, 1]):.3f} vs blend {np.mean(fp[:, 1]):.3f} "
      f"vs GT {np.mean([gt[x['image_name']].mean() for x in rect]):.3f}")

# ---------- 3a. StegaStamp: how close is the U-Net box to the GT rectangle?
errs = []
for x in rect:
    n = x["image_name"]; m = postproc(U[n], False, 0.3, 0)
    lab, k = ndimage.label(m)
    if k == 0:
        errs.append([np.nan] * 4); continue
    big = lab == (1 + np.argmax(ndimage.sum(m, lab, range(1, k + 1))))
    ys, xs = np.where(big); pb = np.array([ys.min(), ys.max() + 1, xs.min(), xs.max() + 1]) * 4
    g = load_mask512(n); gy, gx = np.where(g); gb = np.array([gy.min(), gy.max() + 1, gx.min(), gx.max() + 1])
    errs.append(np.abs(pb - gb))
errs = np.array(errs, dtype=float)
found = ~np.isnan(errs[:, 0])
print(f"[stegastamp] U-Net box vs GT rect: found a region in {found.mean():.0%} of rects; max-edge error median "
      f"{np.nanmedian(errs.max(1)):.0f}px, frac <=8px {np.nanmean(errs.max(1) <= 8):.2f}, <=16px {np.nanmean(errs.max(1) <= 16):.2f}")

# ---------- 3b. StegaStamp decoder tolerance: move ONE edge, or translate, on decodable GT rectangles (CPU ONNX)
from wmloc.watermarks import load_watermark
W = load_watermark("stegastamp", ROOT / "models", device="cpu")


def acc_of(img, box, msg):
    y0, y1, x0, x1 = box
    c = torch.from_numpy(np.ascontiguousarray(img[y0:y1, x0:x1])).permute(2, 0, 1)[None].float() / 255
    c = F.interpolate(c, size=(400, 400), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    return float(((W.logits(c)[0].numpy() > 0) == msg.astype(bool)).mean())


tol = {}
k = 0
for x in [x for x in meta("train") if x["method"] == "stegastamp" and not x["full_watermark"]]:
    g = load_mask512(x["image_name"]); ys, xs = np.where(g); b = [ys.min(), ys.max() + 1, xs.min(), xs.max() + 1]
    if g.sum() / ((b[1] - b[0]) * (b[3] - b[2])) < 0.95:
        continue
    img = load_image("train", x["image_name"]); msg = bits(x["message"])
    if acc_of(img, b, msg) < 0.8:
        continue
    k += 1
    for d in (1, 2, 3, 4, 6, 8):
        for name, bb in (("top", [b[0] + d, b[1], b[2], b[3]]), ("left", [b[0], b[1], b[2] + d, b[3]]),
                         ("bottom", [b[0], b[1] - d, b[2], b[3]]), ("transl", [b[0] + d, b[1] + d, b[2] + d, b[3] + d])):
            bb = [int(np.clip(v, 0, 512)) for v in bb]
            tol.setdefault((name, d), []).append(acc_of(img, bb, msg))
    if k >= 12:
        break
print(f"[stegastamp] decoder tolerance on {k} decodable GT rectangles (mean bit acc; exact box >= 0.8):")
for name in ("top", "left", "bottom", "transl"):
    print("   " + name.ljust(7) + " ".join(f"{d}px:{np.mean(tol[(name, d)]):.2f}" for d in (1, 2, 3, 4, 6, 8)))
