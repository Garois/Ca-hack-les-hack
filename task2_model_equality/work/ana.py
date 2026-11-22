"""Analysis of feat_<TAG>.pt (CPU).

Per endpoint: fitted temperature, quantisation-direction z-scores.
Pairwise: similarity of the endpoints' deviations from the base model, in two feature spaces
  - 'vocab' : global logit-bias score  R_e = sum_t onehot(x_t) - p_t        (Pearson-normalised by expected counts)
  - 'hid'   : hidden-linear score       Phi_e = sum_t g_t z_t^T,  g_t = W^T(onehot - p_t), z_t = [1, PCA_k(h_t)]
For e != f, <S_e, S_f> is unbiased for <mu_e, mu_f> (independent samples); for e = f we use sum over distinct responses.
corr_ef = K_ef / sqrt(K_ee K_ff)   and   z_ef = K_ef / sd_null (sd from per-response inner products).

Usage: python ana.py TAG [--k 16] [--space vocab|hid|both] [--truth]
"""
import argparse

import numpy as np
import torch

import common

torch.set_num_threads(16)


def per_endpoint(meta, eps):
    ep_of = np.array([m["endpoint"] for m in meta])
    return {e: np.where(ep_of == e)[0] for e in eps}


def gram_stats(F, idx, eps):
    """F: (R, P) per-response feature rows (torch float32). Returns K (unbiased), Z (null z), C (corr)."""
    Gm = (F @ F.T).double().numpy()                   # R x R inner products of responses
    n = len(eps)
    K = np.zeros((n, n)); Z = np.zeros((n, n))
    for a, e in enumerate(eps):
        for b, f in enumerate(eps):
            blk = Gm[np.ix_(idx[e], idx[f])]
            if a == b:
                K[a, a] = float(blk.sum() - blk.diagonal().sum())
                Z[a, a] = K[a, a] / float(np.sqrt(2 * (blk ** 2).sum() - 2 * (blk.diagonal() ** 2).sum()) + 1e-9)
            else:
                K[a, b] = float(blk.sum())
                Z[a, b] = K[a, b] / float(np.sqrt((blk ** 2).sum()) + 1e-9)
    d = np.sqrt(np.clip(np.diag(K), 1e-9, None))
    C = K / d[:, None] / d[None, :]
    return K, Z, C


def show(mat, eps, title, fmt="{:6.2f}"):
    print(title)
    print("        " + " ".join(f"{e[-2:]:>6}" for e in eps))
    for a, e in enumerate(eps):
        print(f"  {e} " + " ".join(fmt.format(v) for v in mat[a]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--space", default="both")
    ap.add_argument("--truth", action="store_true")
    ap.add_argument("--minexp", type=float, default=1.0)
    a = ap.parse_args()
    D = torch.load(common.DATA / f"feat_{a.tag}.pt", weights_only=False)
    meta, eps = D["meta"], D["eps"]
    idx = per_endpoint(meta, eps)
    truth = {}
    if a.truth:
        import csv
        for r in csv.DictReader(open(common.ROOT / "practice_answers.csv")):
            truth[r["endpoint"]] = f'{r["model"]}:{r["kind"][:14]}'

    # ---- per-endpoint table
    print(f"== {a.tag}  ({D['model']})")
    sch = list(D["quant"])
    print(f"{'endpoint':9} {'truth':18} {'nresp':>5} {'ntok':>6} {'T':>6}  " + " ".join(f"{s[:11]:>11}" for s in sch))
    for e in eps:
        i = idx[e]
        ntok = sum(meta[j]["ntok"] for j in i)
        zs = []
        for s in sch:
            q = D["quant"][s]
            S, V = q["S"][i], q["V"][i]
            zs.append(f"{(S.sum() / V.sum().sqrt()).item():+5.1f}/{(S.sum() / (S ** 2).sum().sqrt()).item():+5.1f}")
        print(f"{e:9} {truth.get(e, ''):18} {len(i):5d} {ntok:6d} {D['Tfit'][e]:6.3f}  " + " ".join(f"{z:>11}" for z in zs))
    print("  (quant columns: z_theory / z_empirical of the centered score along l_quant - l_base; base ~ 0, quantised > 0)")

    # ---- vocab space
    if a.space in ("vocab", "both"):
        Rv = D["Rv"].float()                       # obs - exp per response
        X, RID = D["X"].long(), D["RID"].long()
        obs = torch.zeros_like(Rv)
        obs.index_put_((RID, X), torch.ones(len(X)), accumulate=True)
        exp = (obs - Rv).clamp(min=0)
        tot_exp = exp.sum(0)
        keepv = tot_exp >= a.minexp * len(eps)      # vocab items with enough expected mass overall
        w = 1.0 / tot_exp[keepv].sqrt()            # Pearson-like normalisation (family-wide expected counts)
        F = Rv[:, keepv] * w
        K, Z, C = gram_stats(F, idx, eps)
        print(f"\n[vocab] kept {int(keepv.sum())} vocab items")
        show(C, eps, "[vocab] corr of deviations (diag = 1 by construction; self-signal z on diag of next table)")
        show(Z, eps, "[vocab] z of shared deviation (null ~ N(0,1)); diag = z of own deviation")

    # ---- hidden-linear space
    if a.space in ("hid", "both"):
        H, Gr, RID = D["H"].float(), D["G"].float(), D["RID"].long()
        mu = H.mean(0)
        sub = torch.randperm(len(H))[:20000]
        U, S_, Vt = torch.linalg.svd(H[sub] - mu, full_matrices=False)
        P = Vt[:a.k].T
        Zf = (H - mu) @ P
        Zf = Zf / Zf.std(0)
        Zf = torch.cat([torch.ones(len(H), 1), Zf], 1)     # [1, k PCs]
        R = len(meta)
        Phi = torch.zeros(R, Gr.shape[1] * Zf.shape[1])
        for r in range(R):
            m = RID == r
            Phi[r] = (Gr[m].T @ Zf[m]).reshape(-1)
        # whiten by the per-coordinate null variance (sum over responses of squares, family-wide)
        Phi = Phi / (Phi ** 2).sum(0).clamp(min=1e-9).sqrt()
        K, Z, C = gram_stats(Phi, idx, eps)
        show(C, eps, f"\n[hid k={a.k}] corr of deviations")
        show(Z, eps, f"[hid k={a.k}] z of shared deviation")


if __name__ == "__main__":
    main()
