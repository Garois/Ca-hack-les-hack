"""Quiz analysis from qcal_<FAM>.json: how typical is each endpoint's quiz answer under the local base model
(T=0.7, and T=1.0), compared with the base's own samples (rank-based p-value, as in RUT), and line-level matching.
Usage: python qana.py FAM"""
import json
import re
import sys
from collections import Counter, defaultdict

import numpy as np

import common

D = json.load(open(common.DATA / f"qcal_{sys.argv[1]}.json"))


def lines(t):
    return [re.sub(r"\s+", " ", l.strip()) for l in t.strip().split("\n") if l.strip()]


def line_ll(resp, ref_samples, alpha=0.5):
    """sum over lines of log empirical frequency of the identical line at the same position in reference samples"""
    tabs = defaultdict(Counter)
    for s in ref_samples:
        for k, l in enumerate(lines(s)):
            tabs[k][l] += 1
    n = len(ref_samples)
    tot = 0.0
    for k, l in enumerate(lines(resp)[:12]):
        tot += np.log((tabs[k][l] + alpha) / (n + alpha * 50))
    return tot


eps = sorted({r["endpoint"] for r in D["endpoint_answers"]})
pids = sorted({r["pid"] for r in D["endpoint_answers"]})
print(f"== {sys.argv[1]}  ({D['model']})")
for pid in pids:
    B = D["samples"][f"{pid}|base"]; F8 = D["samples"][f"{pid}|fp8"]; B1 = D["samples"].get(f"{pid}|base_T1", [])
    llB = np.array([s["ll_base"]["0.7"] for s in B]); llB1 = np.array([s["ll_base"]["1.0"] for s in B1]) if B1 else None
    top = Counter(" | ".join(lines(s["response"])[:3]) for s in B).most_common(3)
    print(f"\n-- {pid}: base T=0.7 most common openings: " + "; ".join(f"{c}x {t[:60]!r}" for t, c in top))
    resp_B = [s["response"] for s in B]; resp_F = [s["response"] for s in F8]
    for r in sorted([r for r in D["endpoint_answers"] if r["pid"] == pid], key=lambda r: r["endpoint"]):
        ll7 = r["ll_base"]["0.7"]; p7 = (llB <= ll7).mean()
        p1 = (llB1 <= r["ll_base"]["1.0"]).mean() if llB1 is not None else float("nan")
        lb, lf = line_ll(r["response"], resp_B), line_ll(r["response"], resp_F)
        print(f"  {r['endpoint']}: ll@0.7={ll7:8.1f} pct_vs_base0.7={p7:5.2f} pct_vs_base1.0={p1:5.2f} | "
              f"lineLL base={lb:6.1f} fp8={lf:6.1f}  ll(fp8)-ll(base)={r['ll_fp8']['0.7'] - ll7:+5.2f} | "
              f"{' | '.join(lines(r['response'])[:3])[:70]!r}")
