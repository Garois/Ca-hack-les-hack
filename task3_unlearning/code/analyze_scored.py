"""Uniform diagnostics for every submission that has a public score -> one JSON line per model on stdout.
Run with --part i --nparts n to split the list over GPUs."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import ANCHORS, ART, DEVICE, ce_vec, kl_vec, load_images_gpu, load_orig, logits_of, tt

W = Path("/p/scratch/training2666/denoncin1/Ca hack les hack/unlearning/lemasnedechermont1")
S, H = W / "submit", Path(os.path.expanduser("~/unl_candidates"))
SCORED = [
    ("P1b_sample", W / "output/P1b_sample/alpha_1.pt", 0.049),
    ("R4_h2", H / "R4_h2_alpha1.pt", 0.010),
    ("R5_all12", H / "R5_all12_alpha1.pt", 0.020),
    ("R9_mu12", S / "R9_s_mu12_b15_alpha1.pt", 0.0544),
    ("R10_mu16", S / "R10_s_mu16_b15_alpha1.pt", 0.0615),
    ("R11_mu22", S / "R11_mu22_a1.pt", 0.069),
    ("R12_mu28", S / "R12_mu28_a1.pt", 0.074),
    ("R12_mu28anc", S / "R12_mu28anc_a1.pt", 0.0651),
    ("R13_mu36", S / "R13_mu36_a1.pt", 0.0784),
    ("R14_or_mu28", S / "R14_or_mu28_a1.pt", 0.0663),
    ("R14_sh33_mu28", S / "R14_sh33_mu28_a1.pt", 0.0522),
    ("R15_or_mu55", S / "R15_or_mu55_a1.pt", 0.0868),
    ("R15_or_mu55anc", S / "R15_or_mu55anc_a1.pt", 0.0788),
]

ap = argparse.ArgumentParser()
ap.add_argument("--part", type=int, default=0)
ap.add_argument("--nparts", type=int, default=1)
args = ap.parse_args()

orig = load_orig()
imgs = load_images_gpu()
y = torch.as_tensor(np.load(ART / "orig_stats.npz")["y"], device=DEVICE).long()
anc = torch.as_tensor(np.load(ANCHORS)["val"], device=DEVICE)
VIEWS = {"clean": ("clean", 0), "held": ("heldout", 4242), "mild": ("mild", 11), "strong": ("strong", 7)}
O = {k: logits_of(orig, imgs, kind=kd, seed=sd) for k, (kd, sd) in VIEWS.items()}
Oa = logits_of(orig, anc)
oh = F.one_hot(y, 100).bool()
runner = O["clean"].masked_fill(oh, -1e9).argmax(1)  # orig's top non-true class


def ent(p):
    return float(-(p * torch.log(p + 1e-30)).sum(-1).mean())


def feats(m):
    r = {}
    for k, (kd, sd) in VIEWS.items():
        z = logits_of(m, imgs, kind=kd, seed=sd)
        ce = ce_vec(z, y)
        p2 = F.softmax(z / 2, -1)
        pred = z.argmax(1)
        wrong = pred != y
        r[f"{k}_ce"] = round(float(ce.mean()), 4)
        r[f"{k}_cesd"] = round(float(ce.std()), 4)
        r[f"{k}_acc"] = round(float((~wrong).float().mean()), 4)
        r[f"{k}_H2"] = round(ent(p2), 4)
        r[f"{k}_pt2"] = round(float(p2[oh].mean()), 4)
        r[f"{k}_kl_orig"] = round(float(kl_vec(O[k], z).mean()), 4)
        r[f"{k}_wrong_runner"] = round(float((pred[wrong] == runner[wrong]).float().mean()) if wrong.any() else 0.0, 4)
    za = logits_of(m, anc)
    pa = F.softmax(za / 2, -1)
    r["anc_kl"] = round(float(kl_vec(Oa, za).mean()), 4)
    r["anc_agree"] = round(float((za.argmax(1) == Oa.argmax(1)).float().mean()), 4)
    r["anc_H2"] = round(ent(pa), 4)
    r["anc_shift_std"] = round(float((za - Oa).mean(0).std()), 4)  # class-level shift magnitude
    return r


print("ORIG " + json.dumps({"name": "orig", **feats(orig)}), flush=True)
for i, (name, path, score) in enumerate(SCORED):
    if i % args.nparts != args.part:
        continue
    if not Path(path).exists():
        print(f"MISSING {name} {path}", flush=True)
        continue
    m = tt.build_model()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    m = m.to(DEVICE).eval()
    print("ROW " + json.dumps({"name": name, "score": score, **feats(m)}), flush=True)
