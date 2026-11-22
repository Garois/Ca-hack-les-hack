"""Collision rate of the first N words between endpoints on the same prompt (near-deterministic openings).
Usage: python firsttok.py N DATA [DATA ...]"""
import sys
from collections import defaultdict

import numpy as np

import common
from fams import FAMILIES

N = int(sys.argv[1])
rows = [r for d in sys.argv[2:] for r in common.load(d)]
by = defaultdict(dict)
for r in rows:
    by[r["id"]][r["endpoint"]] = " ".join(r["response"].split()[:N])
for fam, (m, eps) in FAMILIES.items():
    M = np.zeros((len(eps), len(eps)))
    for i, e in enumerate(eps):
        for j, f in enumerate(eps):
            v = [d[e] == d[f] for d in by.values() if e in d and f in d]
            M[i, j] = np.mean(v)
    print(f"== {fam}: P(first {N} words identical)  (diag = 1)")
    print("        " + " ".join(f"{e[-2:]:>4}" for e in eps))
    for i, e in enumerate(eps):
        print(f"  {e} " + " ".join(f"{v:4.2f}" if i != j else "  - " for j, v in enumerate(M[i])))
