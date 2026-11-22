"""Agreement between quiz answers. Each response is split into lines; item k = k-th line (raw, stripped).
agreement(sample a, sample b) = fraction of the first K lines that are identical.
Prints the endpoint x endpoint mean agreement (diag = within-endpoint agreement across its own samples).
Usage: python quizana.py DATA [--eps a,b,...] [--norm]"""
import argparse
import itertools
import re
from collections import defaultdict

import numpy as np

import common

ap = argparse.ArgumentParser()
ap.add_argument("data", nargs="+")
ap.add_argument("--eps", default=None)
ap.add_argument("--norm", action="store_true")
ap.add_argument("--id", default=None)
a = ap.parse_args()
rows = [r for d in a.data for r in common.load(d) if a.id is None or r["id"] == a.id]
if a.eps:
    keep = a.eps.split(","); rows = [r for r in rows if r["endpoint"] in keep]


def items(t):
    L = [l.strip() for l in t.strip().split("\n") if l.strip()]
    if a.norm:
        L = [re.sub(r"[^a-z0-9]", "", re.sub(r"^\d+\.\s*", "", l.lower())) for l in L]
    return L


S = defaultdict(list)
for r in rows:
    S[r["endpoint"]].append(items(r["response"]))
eps = sorted(S, key=lambda e: (e.split("-")[0], int(e.split("-")[1])))


def agree(x, y):
    K = max(len(x), len(y))
    return sum(1 for i in range(min(len(x), len(y))) if x[i] == y[i]) / K if K else 0


M = np.zeros((len(eps), len(eps)))
for i, e in enumerate(eps):
    for j, f in enumerate(eps):
        if i == j:
            v = [agree(x, y) for x, y in itertools.combinations(S[e], 2)]
        else:
            v = [agree(x, y) for x in S[e] for y in S[f]]
        M[i, j] = np.mean(v) if v else np.nan
print("        " + " ".join(f"{e[-2:]:>4}" for e in eps) + "   (n samples)")
for i, e in enumerate(eps):
    print(f"  {e} " + " ".join(f"{v:4.2f}" for v in M[i]) + f"   {len(S[e])}")
