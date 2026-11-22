"""Split-half reliability of the LoRA-direction projection.
For several prompt splits (dataset-based and random halves of prompt ids) compute, on half A, the anchor deviation
direction mu_E(A) and every endpoint's projection on it using half B (cross-fitted, so noise in the anchor estimate does
not correlate with the endpoint's own noise). Consensus = mean of the presumed-base endpoints (excluded from itself).
Usage: python ana3.py TAG ANCHOR BASESET [--space hid|vocab]"""
import argparse
import numpy as np
import torch
import common
from ana2 import features

ap = argparse.ArgumentParser()
ap.add_argument("tag"); ap.add_argument("anchor"); ap.add_argument("baseset")
ap.add_argument("--space", default="hid"); ap.add_argument("--k", type=int, default=16)
a = ap.parse_args()
D = torch.load(common.DATA / f"feat_{a.tag}.pt", weights_only=False)
meta, eps = D["meta"], D["eps"]
F = features(D, a.space, a.k).double().numpy()
ep_of = np.array([m["endpoint"] for m in meta])
pid = np.array([f'{m.get("data", "")}|{m.get("id", "")}' for m in meta])
print("meta keys:", list(meta[0].keys()))
base = a.baseset.split(",")
uniq = sorted(set(pid))
rng = np.random.default_rng(0)
splits = {}
for s in range(8):
    perm = rng.permutation(len(uniq)); half = set(np.array(uniq)[perm[: len(uniq) // 2]])
    splits[f"rand{s}"] = half


def means(sel):
    return {e: F[(ep_of == e) & sel].mean(0) for e in eps}


res = {e: [] for e in eps}
for name, half in splits.items():
    if half is None:
        dsets = sorted({p.split("|")[0] for p in uniq})
        A = np.array([p.split("|")[0] == "bat_ep" for p in pid])
    else:
        A = np.array([p in half for p in pid])
    for (S1, S2) in ((A, ~A), (~A, A)):
        m1, m2 = means(S1), means(S2)
        c1 = np.mean([m1[e] for e in base if e != a.anchor], 0)
        dE = m1[a.anchor] - c1                       # anchor direction from half 1
        for e in eps:
            c2 = np.mean([m2[b] for b in base if b != e], 0)
            dF = m2[e] - c2
            res[e].append(float(dF @ dE / (dE @ dE)))
print(f"== {a.tag} {a.space} anchor={a.anchor}: cross-fitted alpha over {len(res[eps[0]])} half-splits")
for e in eps:
    v = np.array(res[e])
    tag = "BASESET" if e in base else ""
    print(f"  {e}: mean {v.mean():+.3f}  min {v.min():+.3f}  max {v.max():+.3f}  frac>0 {np.mean(v > 0):.2f} {tag}")
