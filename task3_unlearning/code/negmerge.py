"""NegMerge-style merge (Kim et al., ICML 2025): keep only the coordinates where all task vectors (theta_i - theta_orig)
agree in sign, average them, then scan the extrapolation factor alpha. Evaluates and saves one .pt per alpha."""
import argparse
import copy

import torch

from common import OUT, load_images_gpu, load_orig, save_state_dict
from evaluate_local import Ctx, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--srcs", required=True)
ap.add_argument("--targets", required=True)
ap.add_argument("--alphas", default="1.0,1.2,1.4,1.6")
ap.add_argument("--tag", required=True)
args = ap.parse_args()

orig = load_orig()
th0 = {k: v.detach().float().cpu() for k, v in orig.state_dict().items()}
srcs = args.srcs.split(",")
merged, kept, total = {}, 0, 0
deltas = {k: [] for k in th0}
for p in srcs:
    sd = torch.load(p, map_location="cpu", weights_only=True)
    for k in th0:
        if th0[k].is_floating_point():
            deltas[k].append(sd[k].float() - th0[k])
    del sd
for k, v in th0.items():
    if not v.is_floating_point():
        merged[k] = torch.zeros_like(v)
        continue
    D = torch.stack(deltas[k])
    s = torch.sign(D)
    agree = (s == s[0]).all(0) & (s[0] != 0)
    merged[k] = D.mean(0) * agree
    kept += int(agree.sum())
    total += agree.numel()
    deltas[k] = None
print(f"merged {len(srcs)} task vectors: sign-consensus keeps {kept / total:.1%} of coordinates", flush=True)

imgs = load_images_gpu()
ctx = Ctx(orig, imgs, args.targets, "all")
m = copy.deepcopy(orig)
d = OUT / args.tag
d.mkdir(parents=True, exist_ok=True)
for a in [float(x) for x in args.alphas.split(",")]:
    sd = {k: (th0[k] + a * merged[k]) if th0[k].is_floating_point() else th0[k] for k in th0}
    m.load_state_dict(sd, strict=True)
    r = evaluate(m, ctx)
    c, h = r["clean"], r["heldout_alt"]
    print(f"alpha {a:g}: clean ce {c['ce_mean']:.3f}/{c['ce_std']:.3f} acc {c['acc']:.3f} | held ce {h['ce_mean']:.3f}/"
          f"{h['ce_std']:.3f} | anchor agree {r['anchor']['agree']:.4f} kl {r['anchor']['kl']:.4f} | "
          f"T2 entropy {c['t2_entropy']:.2f}", flush=True)
    save_state_dict(sd, d / f"alpha_{a:g}.pt")
