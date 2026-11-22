"""Shared-rare-answer evidence between endpoints on the quiz prompts.
For each quiz prompt and item k, P_base(answer) is estimated from the local base samples (T=0.7 and T=1.0 pooled,
smoothed). Two endpoints that give the same answer that is rare under the base model share a quirk of their weights.
score(e,f) = sum over prompts/items where both answered identically of  -log P_base(answer)
           minus the expectation of the same quantity for two independent base draws (computed from base samples).
Also prints per-endpoint "atypicality" = mean -log P_base(own answers) relative to base samples.
Usage: python qpair.py FAM"""
import itertools
import json
import re
import sys
from collections import Counter, defaultdict

import numpy as np

import common

D = json.load(open(common.DATA / f"qcal_{sys.argv[1]}.json"))


def items(t):
    L = [l.strip() for l in re.split(r"\n|(?<=\.)\s+(?=\d+\.)", t.strip()) if l.strip()]
    out = {}
    for l in L:
        m = re.match(r"^(\d+)\.\s*(.*)$", l)
        if m:
            k = int(m.group(1))
            v = re.sub(r"^\d+\.\s*", "", m.group(2)).strip().rstrip(".").lower()
            out.setdefault(k, v)
    return out


pids = sorted({r["pid"] for r in D["endpoint_answers"]})
ans = defaultdict(dict)
for r in D["endpoint_answers"]:
    ans[r["endpoint"]][r["pid"]] = items(r["response"])
eps = sorted(ans)
P = {}
for pid in pids:
    base = [items(s["response"]) for s in D["samples"][f"{pid}|base"] + D["samples"].get(f"{pid}|base_T1", [])]
    tab = defaultdict(Counter)
    for b in base:
        for k, v in b.items():
            tab[k][v] += 1
    P[pid] = (tab, len(base), base)


def nlp(pid, k, v):
    tab, n, _ = P[pid]
    return -np.log((tab[k][v] + 0.5) / (n + 0.5 * 20))


# expected shared-surprise for two independent base draws (null)
null = {}
for pid in pids:
    tab, n, base = P[pid]
    tot = 0.0
    for k, c in tab.items():
        for v, cnt in c.items():
            p = cnt / n
            tot += p * p * nlp(pid, k, v)
    null[pid] = tot
print(f"== {sys.argv[1]}: shared rare answers (excess over two independent base draws); rows/cols endpoints")
M = np.zeros((len(eps), len(eps)))
for i, e in enumerate(eps):
    for j, f in enumerate(eps):
        if i >= j:
            continue
        s = 0.0
        for pid in pids:
            if pid in ans[e] and pid in ans[f]:
                a, b = ans[e][pid], ans[f][pid]
                s += sum(nlp(pid, k, a[k]) for k in a if k in b and a[k] == b[k]) - null[pid]
        M[i, j] = M[j, i] = s
print("        " + " ".join(f"{e[-2:]:>5}" for e in eps))
for i, e in enumerate(eps):
    print(f"  {e} " + " ".join(f"{v:5.1f}" if i != j else "    -" for j, v in enumerate(M[i])))
print("atypicality (mean -log P_base per answered item; base samples average in brackets)")
for e in eps:
    vals, ref = [], []
    for pid in ans[e]:
        a = ans[e][pid]
        vals += [nlp(pid, k, v) for k, v in a.items() if k <= 40]
    allb = [nlp(pid, k, v) for pid in pids for b in P[pid][2][:100] for k, v in b.items() if k <= 40]
    print(f"  {e}: {np.mean(vals):5.2f}  [{np.mean(allb):5.2f}]  n_items={len(vals)}")
