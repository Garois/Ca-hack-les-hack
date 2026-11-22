"""Which tokens does each endpoint over/under-use relative to the base model's expectation (at its own temperature)?
Pearson residual per vocab item: z_v = (obs_v - exp_v) / sqrt(exp_v), exp from the base model given the endpoint's own
contexts. Prints top tokens for selected endpoints and the z of those tokens for every endpoint of the family.

Usage: python toptok.py TAG ep-28,ep-03 [--n 20]
"""
import argparse

import numpy as np
import torch
from transformers import AutoTokenizer

import common


def pearson(D):
    Rv = D["Rv"].float()
    X, RID = D["X"].long(), D["RID"].long()
    obs = torch.zeros_like(Rv)
    obs.index_put_((RID, X), torch.ones(len(X)), accumulate=True)
    exp = (obs - Rv).clamp(min=0)
    ep_of = np.array([m["endpoint"] for m in D["meta"]])
    out = {}
    for e in D["eps"]:
        m = torch.tensor(ep_of == e)
        o, x = obs[m].sum(0), exp[m].sum(0)
        out[e] = (o, x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("sel")
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    D = torch.load(common.DATA / f"feat_{a.tag}.pt", weights_only=False)
    tok = AutoTokenizer.from_pretrained(D["model"])
    P = pearson(D)
    eps = D["eps"]
    Z = {e: (o - x) / x.clamp(min=0.5).sqrt() for e, (o, x) in P.items()}
    for s in a.sel.split(","):
        z = Z[s]
        o, x = P[s]
        idx = torch.argsort(z, descending=True)
        print(f"\n### {s}: most OVER-used tokens (obs vs exp) | z of the same token in other endpoints")
        print(" " * 34 + " ".join(f"{e[-2:]:>5}" for e in eps))
        for v in list(idx[:a.n]) + list(idx[-8:]):
            v = int(v)
            print(f"  {tok.decode([v])!r:14} {int(o[v]):5d} {float(x[v]):7.1f} {float(z[v]):+6.1f} |"
                  + " ".join(f"{float(Z[e][v]):+5.1f}" for e in eps))


if __name__ == "__main__":
    main()
