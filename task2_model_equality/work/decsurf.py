"""Profile the (T, top-p) likelihood surface of selected endpoints from dec_<tag>.pt. Usage: python decsurf.py TAG ep,ep"""
import sys

import numpy as np
import torch

import common

D = torch.load(common.DATA / f"dec_{sys.argv[1]}.pt", weights_only=False)
ep_of = np.array([m["endpoint"] for m in D["meta"]])
TS, PS = D["TS"], D["PS"]
for e in sys.argv[2].split(","):
    m = torch.tensor(ep_of == e)
    L = D["llP"][m].sum(0).numpy(); O = D["outP"][m].sum(0).numpy()
    best = L.max()
    print(f"== {e}: ll - best (rows T, cols P)   [tokens outside nucleus in brackets]")
    print("        " + " ".join(f"{p:>12}" for p in PS))
    for i, T in enumerate(TS):
        if 0.75 <= T <= 1.15:
            print(f"  T={T:.2f} " + " ".join(f"{L[i, j] - best:7.1f}[{int(O[i, j]):3d}]" for j in range(len(PS))))
