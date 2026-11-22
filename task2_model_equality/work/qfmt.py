"""Format signature of quiz answers: endpoint formats vs base-sample format frequencies.
Usage: python qfmt.py FAM"""
import json, re, sys
from collections import Counter, defaultdict
import common

D = json.load(open(common.DATA / f"qcal_{sys.argv[1]}.json"))


def fmt(t):
    t = t.strip()
    lines = [l for l in t.split("\n") if l.strip()]
    f = []
    if not re.match(r"^1\.", t):
        f.append("preamble")
    if len(lines) <= 2 and len(re.findall(r"\b\d+\.", t)) > 5:
        f.append("oneline")
    nums = [re.match(r"^\s*(\d+)\.\s*(\d+)\.\s", l) for l in lines]
    dn = [m for m in nums if m]
    if len(dn) >= 3:
        f.append("N.1." if all(m.group(2) == "1" for m in dn) else ("N.N." if all(m.group(1) == m.group(2) for m in dn) else "N.M."))
    items = [l for l in lines if re.match(r"^\s*\d+\.", l)]
    if items and sum(l.rstrip().endswith(".") for l in items) > 0.7 * len(items):
        f.append("dot")
    if items and sum(l.upper() == l for l in items) > 0.7 * len(items):
        f.append("CAPS")
    if "**" in t:
        f.append("bold")
    return "+".join(f) or "plain"


pids = sorted({r["pid"] for r in D["endpoint_answers"]})
for pid in pids:
    print(f"## {pid}")
    for kind in ("base", "fp8", "base_T1"):
        c = Counter(fmt(s["response"]) for s in D["samples"][f"{pid}|{kind}"])
        print(f"  {kind:8}", ", ".join(f"{k}:{v}" for k, v in c.most_common(8)))
    by = defaultdict(list)
    for r in D["endpoint_answers"]:
        if r["pid"] == pid:
            by[fmt(r["response"])].append(r["endpoint"][-2:])
    print("  endpoints:", " | ".join(f"{k}: {','.join(sorted(v))}" for k, v in sorted(by.items())))
