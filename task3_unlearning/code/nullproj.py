"""Null-space projection of a task vector (AlphaEdit / GPM style) to reduce drift on non-forget inputs.
For every nn.Linear (attn.qkv, attn.proj, mlp.fc1, mlp.fc2, head) collect input activations of the ORIGINAL model on the
anchor photos (train split; CLS + ntok random tokens per image), build C = X^T X, keep the top eigenvectors U covering
eps of the energy, and project the weight change: dW <- dW - (dW U) U^T, so outputs on anchor-like inputs stay put.
Other parameters (biases, norms, LayerScale, embeddings) keep their change unless --zero_other 1.
Then scan alpha; evaluate on the forget set + held-out anchors (val split); save one .pt per (eps, alpha)."""
import argparse
import copy

import numpy as np
import torch

from common import ANCHORS, DEVICE, OUT, load_images_gpu, load_orig, norm, save_state_dict, u8_to01
from evaluate_local import Ctx, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--targets", required=True)
ap.add_argument("--eps", default="0.9,0.99")
ap.add_argument("--alphas", default="1.0,1.2,1.4")
ap.add_argument("--ntok", type=int, default=32)
ap.add_argument("--zero_other", type=int, default=0)
ap.add_argument("--tag", required=True)
ap.add_argument("--folds", default="all", choices=["all", "80"])
ap.add_argument("--save", type=int, default=1)
args = ap.parse_args()

orig = load_orig()
lin = {n: mod for n, mod in orig.named_modules() if isinstance(mod, torch.nn.Linear)}
C = {n: torch.zeros(mod.in_features, mod.in_features, dtype=torch.float64, device=DEVICE) for n, mod in lin.items()}
g = torch.Generator(device=DEVICE)
g.manual_seed(0)


def hook(name):
    def f(mod, inp):
        x = inp[0].detach()
        if x.dim() == 3:  # (B, T, d): CLS + ntok random tokens
            B, T, d = x.shape
            idx = torch.cat([torch.zeros(1, dtype=torch.long, device=DEVICE),
                             torch.randint(1, T, (args.ntok,), device=DEVICE, generator=g)])
            x = x[:, idx].reshape(-1, d)
        x = x.double()
        C[name] += x.T @ x
    return f


hs = [mod.register_forward_pre_hook(hook(n)) for n, mod in lin.items()]
anc = torch.as_tensor(np.load(ANCHORS)["train"], device=DEVICE)
with torch.no_grad():
    for i in range(0, len(anc), 64):
        with torch.autocast("cuda", dtype=torch.float16):
            orig(norm(u8_to01(anc[i:i + 64])))
for h in hs:
    h.remove()
eig = {}
for n in lin:
    w, V = torch.linalg.eigh(C[n])
    eig[n] = (w.flip(0).clamp_min(0), V.flip(1))
del C

th0 = {k: v.detach().float() for k, v in orig.state_dict().items()}
src = {k: v.to(DEVICE).float() for k, v in torch.load(args.src, map_location="cpu", weights_only=True).items()}
imgs = load_images_gpu()
ctx = Ctx(orig, imgs, args.targets, args.folds)
m = copy.deepcopy(orig)
d = OUT / args.tag
d.mkdir(parents=True, exist_ok=True)
for e in [float(x) for x in args.eps.split(",")]:
    delta = {}
    for k in th0:
        if not th0[k].is_floating_point():
            continue
        dk = src[k] - th0[k]
        mod_name = k[:-len(".weight")] if k.endswith(".weight") else None
        if mod_name in eig:
            w, V = eig[mod_name]
            frac = torch.cumsum(w, 0) / w.sum()
            kk = int((frac < e).sum()) + 1
            U = V[:, :kk].float()
            dk = dk - (dk @ U) @ U.T
        elif args.zero_other:
            dk = torch.zeros_like(dk)
        delta[k] = dk
    kept = sum(float(delta[k].norm() ** 2) for k in delta) / sum(float((src[k] - th0[k]).norm() ** 2) for k in delta)
    print(f"eps {e:g}: projected task vector keeps {kept:.1%} of its squared norm", flush=True)
    for a in [float(x) for x in args.alphas.split(",")]:
        sd = {k: (th0[k] + a * delta[k]) if k in delta else th0[k] for k in th0}
        m.load_state_dict(sd, strict=True)
        r = evaluate(m, ctx)
        c, hh = r["clean"], r["heldout_alt"]
        print(f"eps {e:g} alpha {a:g}: clean ce {c['ce_mean']:.3f}/{c['ce_std']:.3f} acc {c['acc']:.3f} | held ce "
              f"{hh['ce_mean']:.3f}/{hh['ce_std']:.3f} | anchor agree {r['anchor']['agree']:.4f} kl {r['anchor']['kl']:.4f}"
              f" | T2 entropy {c['t2_entropy']:.2f}"
              + (f" | foldout agree {r['foldout']['agree']:.4f} kl {r['foldout']['kl']:.4f}" if "foldout" in r else ""),
              flush=True)
        if args.save:
            save_state_dict(sd, d / f"eps{e:g}_alpha_{a:g}.pt")
