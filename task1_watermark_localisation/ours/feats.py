"""Message-aware input features for the segmentation net, for real (train/test) and synthetic images:
  map  : 128x128 shift-and-stitch per-position message agreement (pixelseal, trustmark; zeros for stegastamp)
  glob : [mean signed logit, bit accuracy] of the global decoder output (regime evidence)
    run.sh feats.py --method pixelseal
-> cache/feats_<method>.npz with arrays names, maps (N x 128 x 128 float16), glob (N x 2)."""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import CACHE, ROOT, bits, load_image, meta
from common import wm as load_wm
from maps import K, SHIFTS, translate


def sources(method, synth_dir="synth2"):
    for split in ("train", "test"):
        for x in meta(split):
            if x["method"] == method:
                yield x["image_name"], (lambda s=split, n=x["image_name"]: load_image(s, n)), bits(x["message"])
    from synth import iter_shards
    for imgs, m in iter_shards(method, synth_dir):
        for j, n in enumerate(m["names"].tolist()):
            yield n, (lambda imgs=imgs, j=j: np.asarray(imgs[j])), bits(str(m["message"][j]))


@torch.no_grad()
def batch_feats(W, method, imgs, msgs):
    x = torch.stack([torch.from_numpy(i).permute(2, 0, 1).float() / 255 for i in imgs])
    s = torch.from_numpy(np.stack(msgs).astype(np.float32) * 2 - 1)
    lg = W.logits(x if method == "stegastamp" else x.cuda()).float().cpu() * s
    glob = torch.stack([lg.mean(1), (lg > 0).float().mean(1)], 1).numpy()
    if method == "stegastamp":
        return np.zeros((len(imgs), 128, 128), np.float16), glob
    x = x.cuda(); s = s.cuda(); H = None
    for a, dy in enumerate(SHIFTS):
        for b, dx in enumerate(SHIFTS):
            m = (W.features(translate(x, dy, dx))["per_position_logits"].float() * s[:, :, None, None]).mean(1)
            if H is None:
                H = torch.zeros(len(imgs), m.shape[1] * K, m.shape[2] * K, device=m.device)
            H[:, a::K, b::K] = m
    maps = F.interpolate(H[:, None], size=(128, 128), mode="bilinear", align_corners=False)[:, 0]
    return maps.cpu().numpy().astype(np.float16), glob


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--synth_dir", default="synth2"); ap.add_argument("--name", default="feats2")
    a = ap.parse_args()
    W = load_wm(a.method)
    names, maps, globs, buf = [], [], [], []

    def flush():
        m, g = batch_feats(W, a.method, [b[1]() for b in buf], [b[2] for b in buf])
        names.extend(b[0] for b in buf); maps.append(m); globs.append(g); buf.clear()

    for item in sources(a.method, a.synth_dir):
        buf.append(item)
        if len(buf) == a.bs:
            flush()
            if len(names) % 1024 == 0:
                print(a.method, len(names), flush=True)
    if buf:
        flush()
    np.savez(CACHE / f"{a.name}_{a.method}.npz", names=np.array(names), maps=np.concatenate(maps), glob=np.concatenate(globs))
    print("done", a.method, len(names))


if __name__ == "__main__":
    main()
