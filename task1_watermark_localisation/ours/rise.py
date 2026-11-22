"""Randomised-occlusion evidence map for PixelSeal (RISE-style regression).
For each image, N random binary masks over a grid of CELL px cells (p = 0.5): masked-out cells are replaced by a
sigma=6 blur (which removes PixelSeal) and the known-message agreement A = mean_k s_k logit_k is measured.
Ridge regression A ~ b0 + sum_p w_p M_p gives each cell's contribution w_p (holes: ~0, watermarked: > 0).
Two grids offset by CELL/2 are averaged -> CELL/2 effective resolution. Output: 128x128 map per image.
    GPU=k run.sh rise.py --split train --shard k --nshard 4      -> ~/wm_out/rise/rise_<split>_<k>.npz"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import bits, load_image, meta
from common import wm as load_wm
from exp_occl import agreement, blur_rgb

OUT = Path.home() / "wm_out" / "rise"; OUT.mkdir(parents=True, exist_ok=True)


def rise_map(W, x, s, N=512, cell=32, lam=1.0, gen=None, p_keep=0.5, soft=0.0):
    xb = blur_rgb(x, 6.0)
    G = 512 // cell + 1                                              # one extra cell so the offset grid covers the image
    total = torch.zeros(512, 512, device=x.device)
    for off in (0, cell // 2):
        M = (torch.rand(N, 1, G, G, generator=gen, device=x.device) < p_keep).float()
        Mup = F.interpolate(M, scale_factor=cell, mode="nearest")[:, :, off:off + 512, off:off + 512]
        if soft > 0:                                                  # soften cell borders (fewer artificial edges)
            from exp_ps_maps import gauss
            Mup = gauss(Mup, soft)
        A = []
        for i in range(0, N, 128):
            m = Mup[i:i + 128]
            A.append(agreement(W, m * x + (1 - m) * xb, s))
        A = torch.cat(A).float()
        X = M.view(N, -1)                                            # N x G*G
        Xc = X - X.mean(0, keepdim=True); Ac = A - A.mean()
        w = torch.linalg.solve(Xc.T @ Xc + lam * torch.eye(Xc.shape[1], device=x.device), Xc.T @ Ac)
        wmap = F.interpolate(w.view(1, 1, G, G), scale_factor=cell, mode="nearest")[0, 0, off:off + 512, off:off + 512]
        total += wmap
    return F.avg_pool2d((total / 2)[None, None], 4)[0, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train"); ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1); ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--cell", type=int, default=32); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--tag", default="")
    ap.add_argument("--p_keep", type=float, default=0.5); ap.add_argument("--soft", type=float, default=0.0)
    ap.add_argument("--full_only", action="store_true")
    a = ap.parse_args()
    dev = a.device
    W = load_wm("pixelseal", device=dev); gen = torch.Generator(device=dev); gen.manual_seed(1234 + a.shard)
    items = [x for x in meta(a.split) if x["method"] == "pixelseal" and (not a.full_only or x.get("full_watermark", True))][a.shard::a.nshard]
    if a.limit:
        items = items[:a.limit]
    out = {}; t0 = time.time()
    for k, x in enumerate(items):
        xt = torch.from_numpy(load_image(a.split, x["image_name"])).permute(2, 0, 1)[None].float().to(dev) / 255
        s = torch.from_numpy(bits(x["message"]).astype(np.float32) * 2 - 1).to(dev)
        out[x["image_name"]] = rise_map(W, xt, s, a.N, a.cell, gen=gen, p_keep=a.p_keep, soft=a.soft).cpu().numpy().astype(np.float32)
        if k % 10 == 0:
            print(f"{k}/{len(items)} {time.time() - t0:.0f}s", flush=True)
            np.savez(OUT / f"rise_{a.split}_c{a.cell}{a.tag}_{a.shard}.npz", **out)
    np.savez(OUT / f"rise_{a.split}_c{a.cell}{a.tag}_{a.shard}.npz", **out)
    print("done", len(out), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
