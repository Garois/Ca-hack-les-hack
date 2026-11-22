"""Write a submission CSV from a {family: [[group1 eps], [group2 eps], ...]} assignment.
Usage: python make_sub.py OUT.csv   (edit ASSIGN below)"""
import sys

import common

ASSIGN = {
    "Q15": [["ep-06", "ep-07", "ep-08", "ep-20", "ep-21", "ep-26", "ep-30", "ep-32"], ["ep-22"], ["ep-28"], ["ep-03"]],
    "Q3": [["ep-01", "ep-04", "ep-12", "ep-14", "ep-17", "ep-18", "ep-23", "ep-24", "ep-25", "ep-33"], ["ep-02"]],
    "L1": [["ep-10", "ep-13", "ep-16", "ep-19", "ep-27", "ep-29", "ep-31"], ["ep-15"], ["ep-05"], ["ep-09"], ["ep-11"]],
}

rows = []
for fam, groups in ASSIGN.items():
    for k, g in enumerate(groups):
        for ep in g:
            rows.append((ep, fam, f"{fam}_{k}"))
eps = sorted(r[0] for r in rows)
assert eps == common.EPS, ("missing/duplicate endpoints", set(common.EPS) ^ set(eps))
rows.sort()
with open(sys.argv[1], "w") as f:
    f.write("endpoint,family,model\n")
    for r in rows:
        f.write(",".join(r) + "\n")
print(open(sys.argv[1]).read())
