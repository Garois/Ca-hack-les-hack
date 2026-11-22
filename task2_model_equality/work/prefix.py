"""Mean common-prefix length (in words) between endpoints answering the same prompt. Usage: python prefix.py DATA..."""
import sys
from collections import defaultdict

import numpy as np

import common
from fams import FAMILIES

rows = [r for d in sys.argv[1:] for r in common.load(d)]
by = defaultdict(dict)
for r in rows:
    by[(r["id"], r["repeat"])][r["endpoint"]] = r["response"].split()


def cpl(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


for fam, (m, eps) in FAMILIES.items():
    M = np.zeros((len(eps), len(eps)))
    for i, e in enumerate(eps):
        for j, f in enumerate(eps):
            if i < j:
                v = [cpl(d[e], d[f]) for d in by.values() if e in d and f in d]
                M[i, j] = M[j, i] = np.mean(v) if v else np.nan
    print(f"== {fam}: mean common prefix (words)")
    print("        " + " ".join(f"{e[-2:]:>5}" for e in eps))
    for i, e in enumerate(eps):
        print(f"  {e} " + " ".join(f"{v:5.1f}" for v in M[i]))
