"""Coupled-sampling analysis: endpoints appear to share the sampling seed, so two endpoints with the same weights
(and same decoding) give identical text, and similar models share long prefixes.
For every dataset/prompt id answered by both endpoints of a family: common prefix length (chars) and exact equality.
Usage: python coupled.py [FAM ...]"""
import os, sys
from collections import defaultdict
import numpy as np
import common
from fams import FAMILIES

SETS = ["ep_a", "bat_ep", "bat3_ep", "quizA_ep", "quiz2_ep"]
resp = defaultdict(dict)   # (set,id) -> ep -> response
for s in SETS:
    for r in common.load(s):
        resp[(s, r["id"])][r["endpoint"]] = r["response"]


def cpl(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


fams = sys.argv[1:] or list(FAMILIES)
for f in fams:
    eps = sorted(FAMILIES[f][1])
    n = len(eps)
    for s in SETS:
        keys = [k for k in resp if k[0] == s]
        if not keys:
            continue
        EQ = np.zeros((n, n)); PF = np.zeros((n, n)); CNT = np.zeros((n, n))
        for k in keys:
            R = resp[k]
            for i in range(n):
                for j in range(n):
                    if i != j and eps[i] in R and eps[j] in R:
                        a, b = R[eps[i]], R[eps[j]]
                        CNT[i, j] += 1
                        EQ[i, j] += a == b
                        PF[i, j] += np.log1p(cpl(a, b))
        print(f"== {f} {s}: prompts={len(keys)}  cell = #identical / mean log(1+common prefix chars)")
        print("        " + " ".join(f"{e[-2:]:>8}" for e in eps))
        for i in range(n):
            print(f"  {eps[i]} " + " ".join(
                (f"{int(EQ[i, j]):3d}/{PF[i, j] / max(CNT[i, j], 1):4.1f}" if CNT[i, j] else "     -  ") if i != j else "     .  "
                for j in range(n)))
