"""Weight-space extrapolation of an existing candidate: theta_a = theta_orig + a * (theta_src - theta_orig).
No training. Evaluates each a with evaluate_local (same metrics as unlearn.py) and saves a .pt per a."""
import argparse
import copy
import json

import torch

from common import DEVICE, OUT, load_images_gpu, load_orig, save_state_dict
from evaluate_local import Ctx, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--targets", required=True)
ap.add_argument("--alphas", default="1.25,1.5")
ap.add_argument("--tag", required=True)
ap.add_argument("--save", type=int, default=1)
ap.add_argument("--folds", default="all", choices=["all", "80"])
args = ap.parse_args()

orig = load_orig()
imgs = load_images_gpu()
ctx = Ctx(orig, imgs, args.targets, args.folds)
th0 = {k: v.detach().clone() for k, v in orig.state_dict().items()}
th1 = {k: v.to(DEVICE) for k, v in torch.load(args.src, map_location="cpu", weights_only=True).items()}
m = copy.deepcopy(orig)
d = OUT / args.tag
d.mkdir(parents=True, exist_ok=True)
for a in [float(s) for s in args.alphas.split(",")]:
    sd = {k: th0[k] + a * (th1[k] - th0[k]) if th0[k].is_floating_point() else th1[k] for k in th0}
    m.load_state_dict(sd, strict=True)
    r = evaluate(m, ctx)
    c, h = r["clean"], r["heldout_alt"]
    print(f"alpha {a:g}: clean ce {c['ce_mean']:.3f}/{c['ce_std']:.3f} acc {c['acc']:.3f} | held ce {h['ce_mean']:.3f}/"
          f"{h['ce_std']:.3f} | mix agree {r['mix']['agree']:.4f} kl {r['mix']['kl']:.4f}"
          + (f" | anchor agree {r['anchor']['agree']:.4f} kl {r['anchor']['kl']:.4f}" if "anchor" in r else "")
          + f" | T2 entropy clean {c['t2_entropy']:.2f} held {h['t2_entropy']:.2f} ptrue {c['t2_ptrue']:.3f}"
          + (f" | foldout agree {r['foldout']['agree']:.4f} kl {r['foldout']['kl']:.4f}" if "foldout" in r else ""),
          flush=True)
    if args.save:
        save_state_dict(sd, d / f"alpha_{a:g}.pt")
