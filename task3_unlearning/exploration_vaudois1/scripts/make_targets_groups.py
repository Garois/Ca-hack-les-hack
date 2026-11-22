#!/usr/bin/env python3
"""Cibles par groupe de classes. Groupe A = classes « pleines » du forget set (count >= --cap, 9 classes / 1169 images),
groupe B = les 36 classes partielles. Pour chaque groupe : --muA/--muB (CE moyenne visee) ou 'orig' (= garder la
prediction d'orig, aucun oubli). Masse hors vraie classe = melange orig (propre+mild) / vues fortes / uniforme.
    ../.venv/bin/python -I scripts/make_targets_groups.py --muA 8 --muB orig --tag g9A8_Borig
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
S = np.load(ROOT.parent / "lemasnedechermont1" / "artifacts" / "orig_stats.npz")
ap = argparse.ArgumentParser()
ap.add_argument("--muA", required=True)
ap.add_argument("--muB", required=True)
ap.add_argument("--cap", type=int, default=129)
ap.add_argument("--w_orig", type=float, default=0.6)
ap.add_argument("--w_strong", type=float, default=0.3)
ap.add_argument("--w_unif", type=float, default=0.1)
ap.add_argument("--tag", required=True)
a = ap.parse_args()

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
p_base = 0.5 * p_clean + 0.5 * softmax(S["z_mild"].astype(np.float64)).mean(0)
unif = np.ones((n, 100)) / 99.0
unif[ar, y] = 0
w = np.array([a.w_orig, a.w_strong, a.w_unif])
w = w / w.sum()
rest = w[0] * nontrue(p_base) + w[1] * nontrue(S["p_strong"]) + w[2] * unif
grpA = S["counts"][y] >= a.cap
q = p_clean.copy()  # par defaut : prediction d'orig (aucun oubli)
for m, mu in ((grpA, a.muA), (~grpA, a.muB)):
    if mu == "orig":
        continue
    mu = float(mu)
    py = np.clip(p_base[ar, y], 1e-9, 1 - 1e-12)
    lo, hi = -60.0, 60.0
    for _ in range(200):
        c = (lo + hi) / 2
        ps = np.minimum(1 / (1 + np.exp(-c)), py)
        if -np.log(ps[m]).mean() > mu:
            lo = c
        else:
            hi = c
    ps = np.minimum(1 / (1 + np.exp(-(lo + hi) / 2)), py)
    qm = rest * (1 - ps)[:, None]
    qm[ar, y] = ps
    q[m] = qm[m]
q = np.maximum(q, 1e-8)
q /= q.sum(1, keepdims=True)
ce = -np.log(q[ar, y])
info = {"tag": a.tag, **vars(a), "nA": int(grpA.sum()), "nB": int((~grpA).sum()),
        "ce_A": [float(ce[grpA].mean()), float(ce[grpA].std())], "ce_B": [float(ce[~grpA].mean()), float(ce[~grpA].std())],
        "ce_all": [float(ce.mean()), float(ce.std())]}
out = ROOT / "artifacts"
out.mkdir(exist_ok=True)
np.savez_compressed(out / f"targets_{a.tag}.npz", q=q.astype(np.float32), y=y)
(out / f"targets_{a.tag}.json").write_text(json.dumps(info, indent=2))
print(json.dumps(info))
