"""Weight soup: average several candidate state dicts (same init, fine-tuned), evaluate, save one .pt."""
import argparse
import copy

import torch

from common import OUT, load_images_gpu, load_orig, save_state_dict
from evaluate_local import Ctx, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--srcs", required=True, help="comma-separated .pt paths")
ap.add_argument("--targets", required=True)
ap.add_argument("--tag", required=True)
args = ap.parse_args()

sds = [torch.load(p, map_location="cpu", weights_only=True) for p in args.srcs.split(",")]
avg = {k: (sum(sd[k].float() for sd in sds) / len(sds)) if sds[0][k].is_floating_point() else sds[0][k]
       for k in sds[0]}
orig = load_orig()
imgs = load_images_gpu()
ctx = Ctx(orig, imgs, args.targets, "all")
m = copy.deepcopy(orig)
m.load_state_dict(avg, strict=True)
r = evaluate(m, ctx)
c, h = r["clean"], r["heldout_alt"]
print(f"soup of {len(sds)}: clean ce {c['ce_mean']:.3f}/{c['ce_std']:.3f} acc {c['acc']:.3f} | held ce {h['ce_mean']:.3f}/"
      f"{h['ce_std']:.3f} | anchor agree {r['anchor']['agree']:.4f} kl {r['anchor']['kl']:.4f} | "
      f"T2 entropy {c['t2_entropy']:.2f}", flush=True)
d = OUT / args.tag
d.mkdir(parents=True, exist_ok=True)
save_state_dict(avg, d / "alpha_1.pt")
