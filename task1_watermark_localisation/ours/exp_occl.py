"""Occlusion evidence for PixelSeal: contribution of each image region to the known-message agreement.
For a patch p: drop(p) = A(x) - A(x with p replaced by its blurred version), A = mean_k s_k logit_k.
A watermarked patch contributes (drop > 0); a hole (clean) patch does not (drop ~ 0) -> resolution = patch size,
unlike the per-position map S4 whose receptive field is ~whole image.
    run.sh exp_occl.py --device cpu --blur_test           # which blur sigma removes the watermark?
    run.sh exp_occl.py --device cuda --patch 32 --stride 16 --n_full 83 --n_obj 42"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import CACHE, bits, load_image, load_masks128, meta
from common import wm as load_wm
from exp_ps_maps import auc, gauss
from map_rules import is_full
from synth import iter_shards
from wmloc.utils import miou_image


def agreement(W, x, s, bs=128):
    out = []
    for i in range(0, len(x), bs):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            out.append((W.det(W._resize(x[i:i + bs]))[:, 1:].float() * s).mean(1))
    return torch.cat(out)


def blur_rgb(x, sigma):
    return torch.cat([gauss(x[:, c:c + 1], sigma) for c in range(3)], 1)


def occlusion_map(W, x, s, patch, stride, sigma):
    xb = blur_rgb(x, sigma)
    base = agreement(W, x, s)[0]
    ys = list(range(0, 512 - patch + 1, stride)); xs = ys
    drops = torch.zeros(len(ys), len(xs), device=x.device)
    batch, idx = [], []
    for i, y in enumerate(ys):
        for j, xx in enumerate(xs):
            xo = x.clone(); xo[:, :, y:y + patch, xx:xx + patch] = xb[:, :, y:y + patch, xx:xx + patch]
            batch.append(xo); idx.append((i, j))
            if len(batch) == 64 or (i == len(ys) - 1 and j == len(xs) - 1):
                a = agreement(W, torch.cat(batch), s)
                for (ii, jj), v in zip(idx, a):
                    drops[ii, jj] = base - v
                batch, idx = [], []
    # spread each patch's drop over its pixels (average over covering patches) -> 128 grid
    acc = torch.zeros(1, 1, 512, 512, device=x.device); cnt = torch.zeros_like(acc)
    for i, y in enumerate(ys):
        for j, xx in enumerate(xs):
            acc[..., y:y + patch, xx:xx + patch] += drops[i, j]; cnt[..., y:y + patch, xx:xx + patch] += 1
    m = acc / cnt.clamp(min=1)
    return F.avg_pool2d(m, 4)[0, 0].cpu().numpy(), float(base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu"); ap.add_argument("--blur_test", action="store_true")
    ap.add_argument("--patch", type=int, default=64); ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--n_full", type=int, default=3); ap.add_argument("--n_obj", type=int, default=3)
    ap.add_argument("--n_synth", type=int, default=3)
    a = ap.parse_args()
    W = load_wm("pixelseal", device=a.device); gt = load_masks128()
    if a.blur_test:
        for x in [x for x in meta("train") if x["method"] == "pixelseal" and x["full_watermark"]][:4]:
            xt = torch.from_numpy(load_image("train", x["image_name"])).permute(2, 0, 1)[None].float().to(a.device) / 255
            s = torch.from_numpy(bits(x["message"]).astype(np.float32) * 2 - 1).to(a.device)
            row = [f"{agreement(W, xt, s)[0]:.2f}"] + [f"s{sg}:{agreement(W, blur_rgb(xt, sg), s)[0]:.2f}" for sg in (1, 2, 3, 4, 6)]
            print("agreement full / blurred:", " ".join(row), flush=True)
        return
    base_maps = dict(np.load(CACHE / "maps_pixelseal.npz"))
    items = []
    for k, cond, n in (("full/object", True, a.n_full), ("partial/object", False, a.n_obj)):
        c = 0
        for x in meta("train"):
            if x["method"] != "pixelseal" or x["full_watermark"] != cond:
                continue
            g = gt[x["image_name"]]; small = g if not cond else 1 - g; ys, xs = np.where(small)
            if not len(ys) or small.sum() / ((ys.ptp() + 1) * (xs.ptp() + 1)) > 0.95:
                continue
            items.append((k, x["image_name"], load_image("train", x["image_name"]), bits(x["message"]), g, is_full(x))); c += 1
            if c >= n:
                break
    imgs, mt = next(iter_shards("pixelseal", "synth2")); c = 0
    for j in range(len(mt["names"])):
        if c >= a.n_synth:
            break
        if bool(mt["full"][j]):
            items.append(("synth-full", mt["names"][j], np.asarray(imgs[j]), bits(str(mt["message"][j])), mt["mask128"][j], True)); c += 1
    res = {}; t0 = time.time()
    for i, (k, n, img, msg, g, fp) in enumerate(items):
        xt = torch.from_numpy(img).permute(2, 0, 1)[None].float().to(a.device) / 255
        s = torch.from_numpy(msg.astype(np.float32) * 2 - 1).to(a.device)
        om, base = occlusion_map(W, xt, s, a.patch, a.stride, a.sigma)
        res.setdefault((k, "occl"), []).append((auc(om.ravel(), g.ravel()), om, g, fp))
        if n in base_maps:
            res.setdefault((k, "S4"), []).append((auc(base_maps[n].ravel(), g.ravel()), base_maps[n].astype(np.float32), g, fp))
        print(f"{i + 1}/{len(items)} {time.time() - t0:.0f}s {k} AUC occl={res[(k, 'occl')][-1][0]:.3f}"
              + (f" S4={res[(k, 'S4')][-1][0]:.3f}" if n in base_maps else ""), flush=True)
    for key in sorted(res):
        rows = res[key]
        orc = np.mean([max(miou_image((mp >= t).astype(np.uint8), g) for t in np.percentile(mp, np.arange(1, 100, 2))) for _, mp, g, _ in rows])
        print(f"SUMMARY {key[0]:15s} {key[1]:5s} n={len(rows):3d} AUC={np.nanmean([r[0] for r in rows]):.3f} oracle-mIoU={orc:.4f}")
    np.savez_compressed(__import__("pathlib").Path.home() / "wm_out" / f"exp_occl_p{a.patch}.npz",
                        **{f"{k[0]}__{k[1]}__{i}": r[1] for k, rows in res.items() for i, r in enumerate(rows)})


if __name__ == "__main__":
    main()
