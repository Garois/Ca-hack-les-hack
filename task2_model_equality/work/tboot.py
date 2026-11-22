"""Temperature MLE per endpoint with bootstrap CI over responses (from llg stored in feat files).
Usage: python tboot.py TAG [TAG ...]"""
import sys

import numpy as np
import torch

import common


def tfit(ll, grid):
    j = int(ll.argmax()); T = grid[j]
    if 0 < j < len(ll) - 1:
        y0, y1, y2 = ll[j - 1], ll[j], ll[j + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            T += (grid[1] - grid[0]) * 0.5 * (y0 - y2) / den
    return T


rng = np.random.default_rng(0)
for tag in sys.argv[1:]:
    D = torch.load(common.DATA / f"feat_{tag}.pt", weights_only=False)
    llg, grid = D["llg"].numpy(), D["TGRID"].numpy()
    ep_of = np.array([m["endpoint"] for m in D["meta"]])
    print(f"== {tag}")
    res = []
    for e in D["eps"]:
        L = llg[ep_of == e]
        T = tfit(L.sum(0), grid)
        bs = [tfit(L[rng.integers(0, len(L), len(L))].sum(0), grid) for _ in range(300)]
        res.append((T, e, np.std(bs)))
    for T, e, sd in sorted(res):
        print(f"  {e}: T={T:.4f} +- {sd:.4f}")
