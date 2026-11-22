"""Post-hoc class-bias correction (no training).
Our unlearned models shift whole classes up/down globally (drift on unrelated inputs). Estimate the per-class mean
logit shift (sub - orig) on the public anchor photos (train split) and remove a fraction gamma of it from head.bias.
Evaluates forget-set metrics + anchor drift (val split) for each gamma and saves one .pt per gamma."""
import argparse
import copy

import numpy as np
import torch

from common import ANCHORS, ART, DEVICE, OUT, load_images_gpu, load_orig, logits_of, save_state_dict, tt
from evaluate_local import Ctx, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--targets", required=True)
ap.add_argument("--gammas", default="0.5,1.0")
ap.add_argument("--tag", required=True)
ap.add_argument("--save", type=int, default=1)
args = ap.parse_args()

orig = load_orig()
imgs = load_images_gpu()
ctx = Ctx(orig, imgs, args.targets, "all")
sub = tt.build_model()
sd0 = torch.load(args.src, map_location="cpu", weights_only=True)
sub.load_state_dict(sd0, strict=True)
sub = sub.to(DEVICE).eval()
anc = torch.as_tensor(np.load(ANCHORS)["train"], device=DEVICE)
shift = (logits_of(sub, anc) - logits_of(orig, anc)).mean(0)
shift = (shift - shift.mean()).cpu()
counts = np.load(ART / "orig_stats.npz")["counts"]
fc = torch.as_tensor(counts > 0)
print(f"class shift: forget classes mean {shift[fc].mean():+.3f} | other classes mean {shift[~fc].mean():+.3f} | "
      f"max |shift| {shift.abs().max():.3f}", flush=True)
m = copy.deepcopy(sub)
d = OUT / args.tag
d.mkdir(parents=True, exist_ok=True)
for g in [0.0] + [float(s) for s in args.gammas.split(",")]:
    sd = {k: v.clone() for k, v in sd0.items()}
    sd["head.bias"] = sd["head.bias"] - g * shift
    m.load_state_dict(sd, strict=True)
    r = evaluate(m, ctx)
    c, h = r["clean"], r["heldout_alt"]
    print(f"gamma {g:g}: clean ce {c['ce_mean']:.3f}/{c['ce_std']:.3f} acc {c['acc']:.3f} | held ce {h['ce_mean']:.3f}/"
          f"{h['ce_std']:.3f} | anchor agree {r['anchor']['agree']:.4f} kl {r['anchor']['kl']:.4f} | "
          f"T2 entropy {c['t2_entropy']:.2f}", flush=True)
    if args.save and g > 0:
        save_state_dict(sd, d / f"gamma_{g:g}.pt")
