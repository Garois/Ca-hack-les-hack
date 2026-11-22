"""Elementary checks of the PixelSeal residual (float, no rounding) on a clean COCO image:
1. amplitude of r = embed(x0) - x0 (gray levels), channel equality (achromatic?)
2. does re-embedding reproduce it?  corr(r, embed(xw) - xw)
3. message dependence: corr(r_m, r_m') for two different messages
4. matched filter: corr(bp(xw), bp(r'')) vs corr(bp(x0), bp(r'')), with/without uint8 rounding of xw."""
import numpy as np
import torch

from common import wm as load_wm
from exp_ps_maps import bp
from synth import Coco

W = load_wm("pixelseal", device="cpu"); coco = Coco(); rng = np.random.default_rng(1)


def emb(x, m):
    with torch.no_grad():
        return W.model.embed(x, msgs=torch.tensor(m[None], dtype=torch.float32), is_video=False)["imgs_w"]


def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


for t in range(3):
    x0 = torch.from_numpy(coco.image(rng)[0]).permute(2, 0, 1)[None].float() / 255
    m1 = rng.integers(0, 2, 256); m2 = rng.integers(0, 2, 256)
    xw = emb(x0, m1); r = xw - x0
    print(f"--- image {t}: |r| std {r.std() * 255:.2f} levels, max {r.abs().max() * 255:.1f}; "
          f"channel diff std {(r[:, 0] - r[:, 1]).std() * 255:.3f}")
    r2 = emb(xw, m1) - xw
    print(f"re-embed same msg: corr(r, r'') = {corr(r, r2):.3f}   |r''| std {r2.std() * 255:.2f}")
    rm2 = emb(x0, m2) - x0
    print(f"other msg: corr(r_m1, r_m2) = {corr(r, rm2):.3f}  -> message-specific part = r_m1 - r_m2")
    xq = torch.round(xw * 255) / 255
    for name, xx in (("float", xw), ("uint8", xq)):
        u_w = bp(xx.sum(1, keepdim=True)); u_0 = bp(x0.sum(1, keepdim=True))
        rr = bp((emb(xx, m1) - xx).sum(1, keepdim=True)); rn = bp((emb(xx, m2) - xx).sum(1, keepdim=True))
        print(f"[{name}] corr(bp(xw), bp(r'')) = {corr(u_w, rr):.4f} vs clean corr(bp(x0), bp(r'')) = {corr(u_0, rr):.4f}"
              f" | message-specific: corr(bp(xw), bp(r''_m1 - r''_m2)) = {corr(u_w, rr - rn):.4f} vs clean {corr(u_0, rr - rn):.4f}")
