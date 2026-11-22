#!/usr/bin/env python3
"""Cibles souples q(x) pour le forget set (vaudois1). Meme principe que targets.py (mode « sample ») de
lemasnedechermont1, dont on lit orig_stats.npz (lecture seule), avec en plus :
  * un melange reglable des sources de masse hors vraie classe : sonde OOF (P_cv, 45 classes seulement),
    vues fortes (p_strong, 100 classes), orig propre+mild, et une part UNIFORME (couverture de masse : KL(gold||sub)
    penalise lourdement une masse nulle la ou gold en met),
  * une temperature sur cette masse (--tau > 1 lisse, < 1 affute ; R14_sh33 = affute = pire).
Vraie classe : p*(x) = sigmoid(a - b*u(x)), u = quantile normal du rang de difficulte (--proxy), a resolu pour
que la CE moyenne vaille --mu ; plafonne a la confiance d'orig.
Ecrit artifacts/targets_<tag>.npz (q, y) + .json.  CPU, < 5 s :
    ../.venv/bin/python -I scripts/make_targets.py --mu 1.6 --b 1.5 --w_unif 0.2 --tag mix_u20_mu16
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT.parent / "lemasnedechermont1" / "artifacts" / "orig_stats.npz"

ap = argparse.ArgumentParser()
ap.add_argument("--mu", type=float, required=True)
ap.add_argument("--b", type=float, default=1.5)
ap.add_argument("--proxy", default="cv", help="cv | aug | comb | none | liste de {cv,aug,mild,clean} (rang moyen), ex. cv,aug,mild")
ap.add_argument("--w_cv", type=float, default=0.5)
ap.add_argument("--w_strong", type=float, default=0.5)
ap.add_argument("--w_orig", type=float, default=0.0)
ap.add_argument("--w_unif", type=float, default=0.0)
ap.add_argument("--tau", type=float, default=1.0)
ap.add_argument("--floor", type=float, default=1e-6)
ap.add_argument("--tag", required=True)
a = ap.parse_args()

S = np.load(STATS)
y = S["y"].astype(np.int64)
n = len(y)
ar = np.arange(n)


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def nontrue(p):
    p = p.astype(np.float64).copy()
    p[ar, y] = 0
    return p / np.maximum(p.sum(1, keepdims=True), 1e-30)


p_clean = softmax(S["z_clean"].astype(np.float64))
p_mild = softmax(S["z_mild"].astype(np.float64)).mean(0)
p_base = 0.5 * p_clean + 0.5 * p_mild
unif = np.ones((n, 100)) / 99.0
unif[ar, y] = 0
ws = np.array([a.w_cv, a.w_strong, a.w_orig, a.w_unif], dtype=np.float64)
ws = ws / ws.sum()
rest = ws[0] * nontrue(S["P_cv"]) + ws[1] * nontrue(S["p_strong"]) + ws[2] * nontrue(p_base) + ws[3] * unif
if a.tau != 1.0:
    rest = np.power(np.maximum(rest, 1e-30), 1.0 / a.tau)
    rest[ar, y] = 0
    rest /= rest.sum(1, keepdims=True)

if a.proxy == "none":
    u = np.zeros(n)
else:
    def lsm(z):
        z = z - z.max(-1, keepdims=True)
        return z - np.log(np.exp(z).sum(-1, keepdims=True))

    base = {"cv": S["s_cv"], "aug": S["s_aug"],
            "mild": -(lsm(S["z_mild"].astype(np.float64))[:, ar, y]).mean(0),
            "clean": -lsm(S["z_clean"].astype(np.float64))[ar, y]}
    names = ["cv", "aug"] if a.proxy == "comb" else a.proxy.split(",")
    s = np.mean([np.argsort(np.argsort(base[k])) for k in names], 0)
    rank = np.argsort(np.argsort(s + 1e-9 * ar))
    u = torch.special.ndtri(torch.as_tensor((rank + 0.5) / n, dtype=torch.float64)).numpy()
py_base = np.clip(p_base[ar, y], 1e-6, 1 - 1e-9)


def build(c):
    return np.minimum(1 / (1 + np.exp(-(c - a.b * u))), py_base)


lo, hi = -40.0, 40.0
for _ in range(200):
    mid = (lo + hi) / 2
    if -np.log(build(mid)).mean() > a.mu:
        lo = mid
    else:
        hi = mid
ps = build((lo + hi) / 2)
q = rest * (1 - ps)[:, None]
q[ar, y] = ps
q = np.maximum(q, a.floor)
q /= q.sum(1, keepdims=True)
ce = -np.log(q[ar, y])
counts = S["counts"]
wrong = q.argmax(1) != y
info = {"tag": a.tag, **vars(a), "ce_mean": float(ce.mean()), "ce_std": float(ce.std()),
        "acc": float((~wrong).mean()),
        "rest_entropy": float(-(rest * np.log(np.maximum(rest, 1e-30))).sum(1).mean()),
        "rest_mass_on_absent_classes": float(rest[:, counts == 0].sum(1).mean()),
        "argmax_wrong_in_forget_classes": float((counts[q.argmax(1)] > 0)[wrong].mean()) if wrong.any() else None}
out = ROOT / "artifacts"
out.mkdir(exist_ok=True)
np.savez_compressed(out / f"targets_{a.tag}.npz", q=q.astype(np.float32), y=y)
(out / f"targets_{a.tag}.json").write_text(json.dumps(info, indent=2))
print(json.dumps(info))
