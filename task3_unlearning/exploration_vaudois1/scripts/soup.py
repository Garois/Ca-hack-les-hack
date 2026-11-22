#!/usr/bin/env python3
"""Moyenne ponderee de state dicts (« model soup ») : plus robuste sur le split prive de 70 %.
    ../.venv/bin/python -I scripts/soup.py out.pt runs/A/alpha_1.pt runs/B/alpha_1.pt [--w 0.5,0.5]
CPU, ~10 s. Les poids sont normalises ; tout reste en float32."""
import argparse

import torch

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("inputs", nargs="+")
ap.add_argument("--w", default="")
a = ap.parse_args()
w = [float(x) for x in a.w.split(",")] if a.w else [1.0] * len(a.inputs)
assert len(w) == len(a.inputs)
s = sum(w)
w = [x / s for x in w]
acc = None
for wi, p in zip(w, a.inputs):
    sd = torch.load(p, map_location="cpu", weights_only=True)
    if acc is None:
        acc = {k: (v.float() * wi if v.is_floating_point() else v.clone()) for k, v in sd.items()}
    else:
        for k, v in sd.items():
            if v.is_floating_point():
                acc[k] += v.float() * wi
torch.save({k: v.contiguous() for k, v in acc.items()}, a.out)
print("soup ->", a.out, "poids", [round(x, 3) for x in w])
