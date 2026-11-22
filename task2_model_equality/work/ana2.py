"""Deviation analysis v2 (CPU): hidden-linear score space with the family consensus removed (leave-pair-out),
plus projection of every endpoint on an anchor endpoint's deviation (LoRA strength estimate).

  Phi_r = sum_{t in r} g_t z_t^T  (g_t = W^T(onehot - p_t), z_t = [1, k PCs of h_t]) , whitened per coordinate.
  S_e = sum_{r in e} Phi_r ;  for a pair (a,b) the consensus c = sum_{g not in {a,b}} S_g is projected out of both.
  K_ab (a != b) is unbiased for N_a N_b <mu_a, mu_b>; K_aa = sum_{r != r'} -> N_a (N_a - 1) |mu_a|^2.
  alpha_F = <mu_F, mu_E> / |mu_E|^2   (E = anchor), strength proxy s_F / s_E ~ alpha_F * T_F / T_E

Usage: python ana2.py TAG [--k 16] [--anchor ep-28] [--noproj] [--space hid|vocab] [--truth]
"""
import argparse
import csv

import numpy as np
import torch

import common

torch.set_num_threads(16)


def features(D, space, k):
    if space in ("bigram", "uniK", "bigram_c"):
        Rb = D["Rb"].float()                                   # R x P x K
        if space == "uniK":
            F = Rb.sum(1)
        elif space == "bigram_c":                              # remove the prev-independent part (unigram) per response
            F = (Rb - Rb.mean(1, keepdim=True)).reshape(len(Rb), -1)
        else:
            F = Rb.reshape(len(Rb), -1)
        return F / (F ** 2).sum(0).clamp(min=1e-9).sqrt()
    if space == "vocab":
        Rv = D["Rv"].float()
        X, RID = D["X"].long(), D["RID"].long()
        obs = torch.zeros_like(Rv)
        obs.index_put_((RID, X), torch.ones(len(X)), accumulate=True)
        tot_exp = (obs - Rv).clamp(min=0).sum(0)
        keepv = tot_exp >= len(D["eps"])
        F = Rv[:, keepv] / tot_exp[keepv].sqrt()
        return F
    H, Gr, RID = D["H"].float(), D["G"].float(), D["RID"].long()
    mu = H.mean(0)
    g = torch.Generator().manual_seed(0)
    sub = torch.randperm(len(H), generator=g)[:20000]
    _, _, Vt = torch.linalg.svd(H[sub] - mu, full_matrices=False)
    Z = (H - mu) @ Vt[:k].T
    Z = Z / Z.std(0)
    Z = torch.cat([torch.ones(len(H), 1), Z], 1)
    R = int(RID.max()) + 1
    Phi = torch.zeros(R, Gr.shape[1] * Z.shape[1])
    order = torch.argsort(RID)
    bounds = torch.searchsorted(RID[order], torch.arange(R + 1))
    for r in range(R):
        m = order[bounds[r]:bounds[r + 1]]
        Phi[r] = (Gr[m].T @ Z[m]).reshape(-1)
    return Phi / (Phi ** 2).sum(0).clamp(min=1e-9).sqrt()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--space", default="hid")
    ap.add_argument("--anchor", default=None)
    ap.add_argument("--noproj", action="store_true")
    ap.add_argument("--truth", action="store_true")
    ap.add_argument("--base-set", default=None, help="comma list of known-base endpoints used for the consensus")
    a = ap.parse_args()
    D = torch.load(common.DATA / f"feat_{a.tag}.pt", weights_only=False)
    meta, eps, Tfit = D["meta"], D["eps"], D["Tfit"]
    ep_of = np.array([m["endpoint"] for m in meta])
    idx = {e: np.where(ep_of == e)[0] for e in eps}
    truth = {}
    if a.truth:
        for r in csv.DictReader(open(common.ROOT / "practice_answers.csv")):
            truth[r["endpoint"]] = f'{r["model"]}:{r["kind"][:12]}'
    F = features(D, a.space, a.k).double()
    Gm = (F @ F.T).numpy()
    S = torch.stack([F[idx[e]].sum(0) for e in eps]).numpy()          # endpoint sums
    n = len(eps)
    K = np.zeros((n, n)); Z = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            rest = [q for q in range(n) if q not in (i, j)]
            if a.base_set:
                bs = set(a.base_set.split(","))
                rest = [q for q in rest if eps[q] in bs]
            if a.noproj:
                u = None
            else:
                c = S[rest].sum(0)
                u = c / (np.linalg.norm(c) + 1e-12)
            Fi = F[idx[eps[i]]].numpy(); Fj = F[idx[eps[j]]].numpy()
            blk = Gm[np.ix_(idx[eps[i]], idx[eps[j]])]
            if u is not None:
                blk = blk - np.outer(Fi @ u, Fj @ u)
            if i == j:
                k_ = blk.sum() - np.trace(blk)
                sd = np.sqrt(2 * ((blk ** 2).sum() - (np.diag(blk) ** 2).sum()))
            else:
                k_ = blk.sum()
                sd = np.sqrt((blk ** 2).sum())
            K[i, j] = K[j, i] = k_
            Z[i, j] = Z[j, i] = k_ / (sd + 1e-12)
    N = np.array([len(idx[e]) for e in eps], float)
    M = K / np.outer(N, N)                                            # <mu_i, mu_j>
    M[np.diag_indices(n)] = np.diag(K) / (N * (N - 1))                # |mu_i|^2 (unbiased)
    print(f"== {a.tag} space={a.space} k={a.k} consensus_removed={not a.noproj}")
    print("z of shared deviation (diag: own deviation)")
    print("          " + " ".join(f"{e[-2:]:>6}" for e in eps))
    for i, e in enumerate(eps):
        print(f"  {e:6} " + " ".join(f"{v:6.1f}" for v in Z[i]) + f"   T={Tfit[e]:.3f} {truth.get(e, '')}")
    anchors = a.anchor.split(",") if a.anchor else [eps[int(np.argmax(np.diag(Z)))]]
    for E in anchors:
        e0 = eps.index(E)
        print(f"projection on anchor {E} (own z={Z[e0, e0]:.1f}):  alpha = <mu_F,mu_E>/|mu_E|^2 , s~alpha*T/T_E")
        for i, e in enumerate(eps):
            al = M[i, e0] / M[e0, e0]
            print(f"  {e:6} alpha={al:+6.2f}  s_rel={al * Tfit[e] / Tfit[E]:+6.2f}  z={Z[i, e0]:+6.1f}  {truth.get(e, '')}")


if __name__ == "__main__":
    main()
