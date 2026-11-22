"""P1 of PLAN_PIXELSEAL.md — which PixelSeal evidence map localises full-scale watermark presence?
Signals (all on the official 128x128 grid, high = watermarked):
  S1 grad  : gradient x input. G = sum over RGB of d(sum_k s_k logit_k)/dx (PixelSeal's residual is achromatic),
             times the band-passed luma u = R+G+B (DoG 1..6 px: the residual is smooth at 2 px), pooled 16 px.
             Null-normalised: the same with K random messages -> z = (true - mean_null) / std_null.
  S2 reemb : float re-embedding matched filter: bp(u) x bp(r'_Y) with r' = model.embed(x, m) - x (no uint8
             rounding), pooled; null-normalised with K random messages.
  S3 det   : detection channel 0 of the per-position outputs (message-agnostic), 2x2 shift-and-stitch.
  S4 msgmap: current baseline (cached 32x32 shift-and-stitch message map).
Per sub-type: mean per-image AUC, mIoU with the regime-aware relative-threshold rule (q grid fitted on these images,
same protocol for every map) and oracle mIoU (best per-image threshold) as an upper bound.
    run.sh exp_ps_maps.py --device cpu --n_full 24 --n_obj 16 --n_rect 6 --n_synth 12"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from common import CACHE, ROOT, bits, load_image, load_mask512, load_masks128, meta
from common import wm as load_wm
from map_rules import is_full
from synth import iter_shards
from wmloc.utils import miou_image


def gauss(t, s):
    k = int(3 * s) * 2 + 1; x = torch.arange(k, dtype=t.dtype, device=t.device) - k // 2
    w = torch.exp(-x ** 2 / (2 * s * s)); w = w / w.sum()
    t = F.conv2d(F.pad(t, (k // 2, k // 2, 0, 0), mode="reflect"), w.view(1, 1, 1, k))
    return F.conv2d(F.pad(t, (0, 0, k // 2, k // 2), mode="reflect"), w.view(1, 1, k, 1))


def bp(t):                                   # band-pass keeping ~2..12 px structures at 512
    return gauss(t, 1.0) - gauss(t, 6.0)


def pool(t, win=16):                         # -> 128x128 grid (cell = 4 px), local window win px
    return F.avg_pool2d(F.pad(t, ((win - 4) // 2,) * 4, mode="reflect"), win, stride=4)[..., :128, :128]


class PS:
    def __init__(self, device):
        self.W = load_wm("pixelseal", device=device); self.dev = device
        self.det = self.W.det; self.model = self.W.model

    def logits_full(self, x):                # x: B x 3 x 512 x 512 in [0,1] -> B x 257 (channel 0 = detection)
        return self.det(self.W._resize(x))

    def grad_map(self, x, s):
        x = x.clone().requires_grad_(True)
        lg = self.logits_full(x)[:, 1:]
        (lg * s).sum().backward()
        G = x.grad.sum(1, keepdim=True)                                   # achromatic direction
        u = x.detach().sum(1, keepdim=True)
        return pool(G * bp(u))

    def reemb_c_maps(self, x, msg, wins=(16, 32)):
        """Matched filter on the MESSAGE-SPECIFIC residual d = r(x, m) - r(x, 1-m): the content-driven part common to
        all messages cancels (it was anti-correlated with the content and drowned the signal)."""
        with torch.no_grad():
            m = msg[None].float().to(self.dev)
            r1 = (self.model.embed(x, msgs=m, is_video=False)["imgs_w"] - x).sum(1, keepdim=True)
            r0 = (self.model.embed(x, msgs=1 - m, is_video=False)["imgs_w"] - x).sum(1, keepdim=True)
            a, b = bp(x.sum(1, keepdim=True)), bp(r1 - r0)
            return {w: (pool(a * b, w) / torch.sqrt(pool(a * a, w) * pool(b * b, w) + 1e-12))[0, 0].cpu().numpy() for w in wins}

    def reemb_map(self, x, msg):
        with torch.no_grad():
            out = self.model.embed(x, msgs=msg[None].float().to(self.dev), is_video=False)
            r = (out["imgs_w"] - x).sum(1, keepdim=True)
            u = x.sum(1, keepdim=True)
            return pool(bp(u) * bp(r))

    @torch.no_grad()
    def det_map(self, x):
        H = torch.zeros(1, 1, 16, 16, device=x.device)
        for a, dy in enumerate((-16, 16)):
            for b, dx in enumerate((-16, 16)):
                xp = F.pad(x, (32,) * 4, mode="reflect")[:, :, 32 + dy:32 + dy + 512, 32 + dx:32 + dx + 512]
                f = self.det.preprocess(self.W._resize(xp)); lat = self.det.convnext(f)
                emb = self.det.pixel_decoder.output_upscaling(lat); lin = self.det.pixel_decoder.linear
                pp = torch.einsum("bchw,oc->bohw", emb, lin.weight[:1]) + lin.bias[None, :1, None, None]
                H[:, :, a::2, b::2] = pp
        return F.interpolate(H, size=(128, 128), mode="bilinear", align_corners=False)


def znull(true, nulls):
    n = torch.stack(nulls); return ((true - n.mean(0)) / (n.std(0) + 1e-8))[0, 0]


def auc(score, lab):
    pos, neg = score[lab == 1], score[lab == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = np.argsort(np.argsort(np.concatenate([pos, neg])))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))


def rule(m, full, q, smooth):
    if smooth:
        m = ndimage.gaussian_filter(m, smooth)
    lo, hi = np.percentile(m, 5), np.percentile(m, 99)
    t = lo + q * (hi - lo)
    return (m >= t).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu"); ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--n_full", type=int, default=24); ap.add_argument("--n_obj", type=int, default=16)
    ap.add_argument("--n_rect", type=int, default=6); ap.add_argument("--n_synth", type=int, default=12)
    ap.add_argument("--out", default="exp_ps_maps.npz")
    a = ap.parse_args()
    torch.manual_seed(0); rng = np.random.default_rng(0)
    ps = PS(a.device); gt = load_masks128(); base = dict(np.load(CACHE / "maps_pixelseal.npz"))
    # sample: real train by sub-type, plus synth2 (aligned by construction) as a sanity check
    sel = {"full/object": [], "partial/object": [], "partial/rect": []}
    for x in meta("train"):
        if x["method"] != "pixelseal":
            continue
        g = gt[x["image_name"]]; small = g if not x["full_watermark"] else 1 - g
        ys, xs = np.where(small)
        shape = "rect" if len(ys) and small.sum() / ((ys.ptp() + 1) * (xs.ptp() + 1)) > 0.95 else "object"
        k = ("full" if x["full_watermark"] else "partial") + "/" + shape
        if k in sel:
            sel[k].append(x)
    items = []
    for k, n in (("full/object", a.n_full), ("partial/object", a.n_obj), ("partial/rect", a.n_rect)):
        for x in sel[k][:n]:
            items.append((k, "real", x["image_name"], load_image("train", x["image_name"]), bits(x["message"]), gt[x["image_name"]], is_full(x)))
    imgs, m = next(iter_shards("pixelseal", "synth2"))
    sk = 0
    for j in range(len(m["names"])):
        if sk >= a.n_synth:
            break
        full = bool(m["full"][j]); g = m["mask128"][j]
        if not full:
            small = g; ys, xs = np.where(small)
            if not len(ys) or small.sum() / ((ys.ptp() + 1) * (xs.ptp() + 1)) > 0.95:
                continue                      # synth own-scale rectangles are not the target here
        items.append(("synth-" + ("full" if full else "partial/object"), "synth", m["names"][j], np.asarray(imgs[j]),
                      bits(str(m["message"][j])), g, full)); sk += 1
    print(f"{len(items)} images on {a.device}", flush=True)
    res = {}; maps_all = {}
    t0 = time.time()
    for i, (k, src, n, img, msg, g, full_p) in enumerate(items):
        x = torch.from_numpy(img).permute(2, 0, 1)[None].float().to(a.device) / 255
        s = torch.from_numpy(msg.astype(np.float32) * 2 - 1).to(a.device)
        nulls_s = [torch.from_numpy(rng.integers(0, 2, 256).astype(np.float32) * 2 - 1).to(a.device) for _ in range(a.K)]
        S2c = ps.reemb_c_maps(x, torch.from_numpy(msg).to(a.device))
        S3 = ps.det_map(x)[0, 0].cpu().numpy()
        maps = {"S2c16": S2c[16], "S2c32": S2c[32], "S3det": S3}
        if n in base:
            maps["S4msgmap"] = base[n].astype(np.float32)
        maps_all[n] = maps
        for name, mp in maps.items():
            res.setdefault((k, name), []).append((auc(mp.ravel(), g.ravel()), mp, g, full_p))
        if i % 5 == 0:
            print(f"{i}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    print("\nsub-type          map        n   AUC   rule-mIoU (q,smooth)   oracle-mIoU  | current-pipeline-like const")
    for (k, name) in sorted(res):
        rows = res[(k, name)]
        aucs = np.array([r[0] for r in rows], dtype=float)
        best = (-1, None)
        for q in np.linspace(0.05, 0.95, 19):
            for sm in (0, 2):
                sc = np.mean([miou_image(rule(mp, fp, q, sm), g) for _, mp, g, fp in rows])
                if sc > best[0]:
                    best = (sc, (round(float(q), 2), sm))
        orc = np.mean([max(miou_image((mp >= t).astype(np.uint8), g) for t in np.percentile(mp, np.arange(1, 100, 2))) for _, mp, g, _ in rows])
        const = np.mean([miou_image(np.full((128, 128), int(fp), np.uint8), g) for _, mp, g, fp in rows])
        print(f"{k:17s} {name:9s} {len(rows):3d} {np.nanmean(aucs):.3f}  {best[0]:.4f} {best[1]!s:12s}  {orc:.4f}      | {const:.4f}", flush=True)
    np.savez_compressed(ROOT / "ours" / "cache" / a.out if False else __import__("pathlib").Path.home() / "wm_out" / a.out,
                        **{f"{n}__{k}": v for n, d in maps_all.items() for k, v in d.items()})


if __name__ == "__main__":
    main()
