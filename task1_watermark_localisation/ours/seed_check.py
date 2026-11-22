"""B3 entry condition (PLAN_V9 W4): with the new decision layer (calibrated expected-mIoU decoding), how close are the
StegaStamp seed boxes to the GT rectangles? Only DECODABLE GT rectangles (exact-box bit acc >= 0.8) matter.
Seeds tried: bbox of the main decoded region, and bbox of the q >= q_low region containing it."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from common import ROOT, bits, load_image, load_mask512, load_masks128
from decision import apply_iso, emiou_mask, fit_iso, oof_probs
from map_rules import is_full
from ps_probe import subtype
from train_seg import folds
from wmloc.watermarks import load_watermark

gt = load_masks128(); m = "stegastamp"
items, fold_of = folds(m); P = oof_probs(m); names = [x["image_name"] for x in items]; X = {x["image_name"]: x for x in items}
fullp = {n: is_full(X[n]) for n in names}
cal = {}
for f in range(5):
    for full in (True, False):
        tr = [n for n in names if fold_of[n] != f and fullp[n] == full]
        cal[(f, full)] = fit_iso(np.concatenate([P[n].ravel() for n in tr]), np.concatenate([gt[n].ravel() for n in tr]).astype(float))
W = load_watermark("stegastamp", ROOT / "models", device="cpu")


def acc_of(img, b, msg):
    c = torch.from_numpy(np.ascontiguousarray(img[b[0]:b[1], b[2]:b[3]])).permute(2, 0, 1)[None].float() / 255
    c = F.interpolate(c, size=(400, 400), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    return float(((W.logits(c)[0].numpy() > 0) == msg.astype(bool)).mean())


def box_of(reg):
    ys, xs = np.where(reg); return np.array([ys.min(), ys.max() + 1, xs.min(), xs.max() + 1]) * 4


rows = []
for n in names:
    x = X[n]
    if subtype(x, gt[n]) != "partial/rect":
        continue
    g = load_mask512(n); gy, gx = np.where(g); gb = np.array([gy.min(), gy.max() + 1, gx.min(), gx.max() + 1])
    dec = acc_of(load_image("train", n), gb, bits(x["message"])) >= 0.8
    q = apply_iso(cal[(fold_of[n], fullp[n])], P[n]); mk = emiou_mask(q)
    out = [dec]
    for kind in ("main", "low0.15", "low0.3"):
        if not mk.any() or mk.mean() >= 0.5:
            out += [np.nan, np.nan]; continue
        lab, k = ndimage.label(mk); big = lab == (1 + np.argmax(ndimage.sum(mk, lab, range(1, k + 1))))
        reg = big
        if kind != "main":
            low, _ = ndimage.label(q >= float(kind[3:]))
            lid = np.bincount(low[big]).argmax() if (low[big] > 0).any() else 0
            reg = (low == lid) if lid > 0 else big
        b = box_of(reg)
        cerr = max(abs((b[0] + b[1]) / 2 - (gb[0] + gb[1]) / 2), abs((b[2] + b[3]) / 2 - (gb[2] + gb[3]) / 2))
        serr = max(abs((b[1] - b[0]) - (gb[1] - gb[0])), abs((b[3] - b[2]) - (gb[3] - gb[2])))
        out += [cerr, serr]
    rows.append(out)
r = np.array(rows, dtype=float); d = r[:, 0] == 1
print(f"StegaStamp partial/rect: n={len(r)}, decodable (exact box acc >= 0.8): {int(d.sum())}")
for i, kind in enumerate(("main", "low0.15", "low0.3")):
    ce, se = r[d, 1 + 2 * i], r[d, 2 + 2 * i]
    print(f"  seed={kind:8s} among decodable: center err median {np.nanmedian(ce):.0f}px, <=12px {np.nanmean(ce <= 12):.2f}, "
          f"<=6px {np.nanmean(ce <= 6):.2f} | size err median {np.nanmedian(se):.0f}px, <=32px {np.nanmean(se <= 32):.2f} | no seed {np.mean(np.isnan(ce)):.2f}")
