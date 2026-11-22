"""Write a submission CSV from {family: [[eps of model 1], [eps of model 2], ...]}. Usage: from write_sub import write"""
import common

FAMLETTER = {"Q15": "A", "Q3": "B", "L1": "C"}


def write(assign, path):
    rows = []
    for fam, groups in assign.items():
        L = FAMLETTER[fam]
        for k, g in enumerate(groups):
            for ep in g:
                rows.append((ep, L, f"{L}{k + 1}"))
    eps = sorted(r[0] for r in rows)
    assert eps == common.EPS, ("missing/duplicate endpoints", set(common.EPS) ^ set(eps))
    rows.sort()
    with open(path, "w") as f:
        f.write("endpoint,family,model\n")
        for r in rows:
            f.write(",".join(r) + "\n")
    return rows
