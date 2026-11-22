"""Exact-rectangle refinement for the partial regime.
Real partial regions are ~50% rectangles watermarked at their OWN scale (outpainting): decoding exactly that crop
recovers the known message. Starting from a box, hill-climb its 4 edges (16 -> 8 -> 4 -> 2 px steps at 512 res) to
maximise the mean signed logit of the crop; if the final bit accuracy is high, the rectangle is the answer.
    GPU=3 run.sh rect_search.py --method trustmark --eval gtjitter   # sanity check from jittered GT boxes
    GPU=3 run.sh rect_search.py --method trustmark --eval val --probs ckpt/trustmark_f0_aux_valprobs.npz"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from common import ROOT, bits, load_image, load_mask512, load_masks128, meta
from common import wm as load_wm
from wmloc.utils import downsample_mask, miou_image

RES = {"pixelseal": (256, "bilinear"), "trustmark": (245, "bilinear"), "stegastamp": (400, "bicubic")}


class Scorer:
    """Batched decoding of variable-size crops: each crop is resized to the decoder's working resolution with the
    same interpolation the wrapper uses, so the wrapper's own resize becomes the identity."""

    def __init__(self, method):
        self.method = method; self.W = load_wm(method); self.res, self.mode = RES[method]

    @torch.no_grad()
    def __call__(self, img_t, boxes, sign):
        """img_t: 1 x 3 x 512 x 512 float cuda; boxes: list of (y0, y1, x0, x1); sign: nbits (+-1) cuda.
        Returns (mean signed logit, bit accuracy) per box."""
        crops = []
        for (y0, y1, x0, x1) in boxes:
            c = img_t[:, :, y0:y1, x0:x1]
            c = F.interpolate(c, size=(self.res, self.res), mode=self.mode, align_corners=False, antialias=True)
            crops.append(c.clamp(0, 1) if self.mode == "bicubic" else c)
        x = torch.cat(crops)
        lg = self.W.logits(x if self.method != "stegastamp" else x.cpu()).float().to(sign.device) * sign
        return lg.mean(1).cpu().numpy(), (lg > 0).float().mean(1).cpu().numpy()


def clip_box(b, min_side=16):
    y0, y1, x0, x1 = [int(v) for v in b]
    y0 = max(0, min(y0, 512 - min_side)); x0 = max(0, min(x0, 512 - min_side))
    y1 = min(512, max(y1, y0 + min_side)); x1 = min(512, max(x1, x0 + min_side))
    return (y0, y1, x0, x1)


def hill_climb(scorer, img_t, sign, box, steps=(16, 8, 4, 2), max_iter=12):
    box = clip_box(box)
    best_s, best_a = scorer(img_t, [box], sign); best_s, best_a = float(best_s[0]), float(best_a[0])
    for st in steps:
        for _ in range(max_iter):
            cands = []
            for k in range(4):
                for d in (-st, st):
                    b = list(box); b[k] += d; b = clip_box(b)
                    if b != box:
                        cands.append(b)
            cands = list(dict.fromkeys(cands))
            s, a = scorer(img_t, cands, sign)
            j = int(np.argmax(s))
            if s[j] <= best_s:
                break
            box, best_s, best_a = cands[j], float(s[j]), float(a[j])
    return box, best_s, best_a


def init_boxes(prob, thr=0.5):
    """Candidate starting boxes (512-px coords) from a 128x128 probability map: bbox of each sizeable component at a
    few thresholds."""
    out = []
    for t in (thr, 0.3, 0.7):
        lab, n = ndimage.label(prob > t)
        for sl in ndimage.find_objects(lab):
            if sl is None:
                continue
            ys, xs = sl
            if (ys.stop - ys.start) * (xs.stop - xs.start) < 6:
                continue
            out.append((ys.start * 4, ys.stop * 4, xs.start * 4, xs.stop * 4))
    return list(dict.fromkeys(out))[:6]


def box_mask(box):
    m = np.zeros((512, 512), np.uint8); y0, y1, x0, x1 = box; m[y0:y1, x0:x1] = 1
    return downsample_mask(m)


def refine(scorer, img, msg, prob, acc_thr=0.8):
    """Returns (mask128 or None, best box, best bit acc)."""
    img_t = torch.from_numpy(img).permute(2, 0, 1)[None].float().cuda() / 255
    sign = torch.from_numpy(msg.astype(np.float32) * 2 - 1).cuda()
    best = (None, -1e9, 0.0)
    for b in init_boxes(prob):
        box, s, a = hill_climb(scorer, img_t, sign, b)
        if s > best[1]:
            best = (box, s, a)
    if best[0] is not None and best[2] >= acc_thr:
        return box_mask(best[0]), best[0], best[2]
    return None, best[0], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--eval", default="gtjitter")
    ap.add_argument("--probs", default=""); ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    sc = Scorer(a.method); gt = load_masks128(); rng = np.random.default_rng(0)
    xs = [x for x in meta("train") if x["method"] == a.method and not x["full_watermark"]]
    if a.eval == "gtjitter":
        rows = []
        for x in xs[:a.n]:
            m = load_mask512(x["image_name"]); ys, xx = np.where(m)
            fill = m.sum() / ((ys.ptp() + 1) * (xx.ptp() + 1))
            g = (ys.min(), ys.max() + 1, xx.min(), xx.max() + 1)
            j = [g[k] + int(rng.integers(-20, 21)) for k in range(4)]
            img = load_image("train", x["image_name"]); msg = bits(x["message"])
            img_t = torch.from_numpy(img).permute(2, 0, 1)[None].float().cuda() / 255
            sign = torch.from_numpy(msg.astype(np.float32) * 2 - 1).cuda()
            box, s, acc = hill_climb(sc, img_t, sign, j)
            rows.append((fill > 0.95, acc, miou_image(box_mask(box), gt[x["image_name"]]), miou_image(box_mask(clip_box(j)), gt[x["image_name"]])))
        r = np.array(rows, dtype=float)
        for is_rect in (1.0, 0.0):
            s = r[r[:, 0] == is_rect]
            print(f"{a.method} {'rect  ' if is_rect else 'object'} n={len(s)} acc med={np.median(s[:, 1]):.3f} "
                  f"frac acc>=.8={np.mean(s[:, 1] >= .8):.2f} | mIoU jitter-init={s[:, 3].mean():.3f} -> refined={s[:, 2].mean():.3f}")
    else:
        from train_seg import ckpt_path
        z = np.load(ckpt_path(a.probs.replace(".npz", "")).with_suffix(".npz") if not a.probs.endswith(".npz") else ROOT / "ours" / a.probs); P = dict(zip(z["names"].tolist(), z["probs"].astype(np.float32)))
        full_of = {x["image_name"]: x["full_watermark"] for x in meta("train")}
        base, new, used = [], [], 0
        for n, p in P.items():
            x = next(x for x in meta("train") if x["image_name"] == n)
            b = miou_image((p > 0.5).astype(np.uint8), gt[n]); base.append(b)
            if full_of[n] or p.mean() > 0.5:          # only the partial regime
                new.append(b); continue
            mk, box, acc = refine(sc, load_image("train", n), bits(x["message"]), p)
            if mk is not None:
                used += 1; new.append(miou_image(mk, gt[n]))
            else:
                new.append(b)
        print(f"{a.method} val n={len(P)}: base mIoU={np.mean(base):.4f} -> with rect refinement={np.mean(new):.4f} (used on {used})")


if __name__ == "__main__":
    main()
