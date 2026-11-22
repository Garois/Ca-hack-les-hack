"""Re-target the head-bias surgery of an existing candidate: --src model, --base (the same model before surgery, or
the src itself with --src_D), sets head.bias[removed classes] so the median clean CE of removed images == --rm_ce."""
import argparse

import numpy as np
import torch

from common import ART, DEVICE, ce_vec, load_images_gpu, logits_of, save_state_dict, tt

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--rm_ce", type=float, required=True, help="0 -> undo surgery entirely (needs --src_D)")
ap.add_argument("--src_D", type=float, default=0.0, help="surgery D already applied in --src")
ap.add_argument("--cap", type=int, default=129)
ap.add_argument("--out", required=True)
args = ap.parse_args()

S = np.load(ART / "orig_stats.npz")
counts = S["counts"]
cls = torch.as_tensor(np.where(counts >= args.cap)[0])
y = torch.as_tensor(S["y"], device=DEVICE).long()
rm = torch.as_tensor(counts[S["y"]] >= args.cap, device=DEVICE)
sd = torch.load(args.src, map_location="cpu", weights_only=True)
sd["head.bias"] = sd["head.bias"].clone()
sd["head.bias"][cls] += args.src_D  # back to pre-surgery
m = tt.build_model()
m.load_state_dict(sd, strict=True)
m = m.to(DEVICE).eval()
imgs = load_images_gpu()
z = logits_of(m, imgs[torch.nonzero(rm).squeeze(1)])
shift = torch.zeros(100, device=DEVICE)
shift[cls.to(DEVICE)] = 1.0
D = 0.0
if args.rm_ce > 0:
    lo, hi = 0.0, 60.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if ce_vec(z - mid * shift, y[rm]).median() < args.rm_ce:
            lo = mid
        else:
            hi = mid
    D = (lo + hi) / 2
sd["head.bias"][cls] -= D
save_state_dict(sd, args.out)
print(f"D={D:.3f} median removed CE={float(ce_vec(z - D * shift, y[rm]).median()):.3f} -> {args.out}")
