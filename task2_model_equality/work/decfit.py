"""Decoding-strategy fit per endpoint: temperature x top-p, temperature x top-k, temperature x min-p, under the local
base model. For each response and grid cell we store the log-likelihood (tokens outside the truncation set get
log(1e-6) instead of -inf) and the number of tokens outside the set.

A base model served with top-p=P has ~0 tokens outside its nucleus; a different model (quantised / LoRA) does not.

Usage: python decfit.py TAG MODEL_ID DATA [DATA ...] [--eps a,b,c]     -> data/dec_<TAG>.pt
"""
import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import common
from feat import encode, fwd

DEV = "cuda"
TS = [round(x, 2) for x in np.arange(0.55, 1.301, 0.05)]
PS = [0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.99, 1.0]
KS = [10, 20, 40, 50, 100, 200, 1000000]
MPS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
LOGEPS = float(np.log(1e-6))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("model")
    ap.add_argument("data", nargs="+")
    ap.add_argument("--eps", default=None)
    a = ap.parse_args()
    keep = set(a.eps.split(",")) if a.eps else None
    rows = [r for d in a.data for r in common.load(d) if keep is None or r["endpoint"] in keep]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, device_map=DEV).eval()
    enc = encode(tok, rows)
    R = len(enc)
    llP = torch.zeros(R, len(TS), len(PS)); outP = torch.zeros(R, len(TS), len(PS))
    llK = torch.zeros(R, len(TS), len(KS)); outK = torch.zeros(R, len(TS), len(KS))
    llM = torch.zeros(R, len(TS), len(MPS)); outM = torch.zeros(R, len(TS), len(MPS))
    t0 = time.time()
    for i, (r, c, x) in enumerate(enc):
        lg, _, xt = fwd(model, c, x)
        srt, order = lg.sort(-1, descending=True)                     # same order for every T
        rank = (order == xt[:, None]).float().argmax(-1)               # rank of x (0 = argmax)
        for a_, T in enumerate(TS):
            lp = torch.log_softmax(srt / T, -1)
            p = lp.exp()
            cum = p.cumsum(-1)
            lpx = lp.gather(1, rank[:, None].long())[:, 0]
            cum_excl = cum.gather(1, rank[:, None].long())[:, 0] - lpx.exp()
            for b, P in enumerate(PS):
                if P >= 1.0:
                    inside = torch.ones_like(lpx, dtype=torch.bool); logM = torch.zeros_like(lpx)
                else:
                    kcut = (cum < P).sum(-1).clamp(max=cum.shape[1] - 1)    # nucleus = indices 0..kcut
                    logM = torch.log(cum.gather(1, kcut[:, None])[:, 0])
                    inside = cum_excl < P
                ll = torch.where(inside, lpx - logM, torch.full_like(lpx, LOGEPS))
                llP[i, a_, b] = ll.sum().cpu(); outP[i, a_, b] = (~inside).sum().cpu()
            for b, K in enumerate(KS):
                kk = min(K, cum.shape[1])
                logM = torch.log(cum[:, kk - 1])
                inside = rank < kk
                ll = torch.where(inside, lpx - logM, torch.full_like(lpx, LOGEPS))
                llK[i, a_, b] = ll.sum().cpu(); outK[i, a_, b] = (~inside).sum().cpu()
            pmax = p[:, 0]
            for b, mp in enumerate(MPS):
                keepm = p >= mp * pmax[:, None]
                logM = torch.log((p * keepm).sum(-1))
                inside = lpx.exp() >= mp * pmax
                ll = torch.where(inside, lpx - logM, torch.full_like(lpx, LOGEPS))
                llM[i, a_, b] = ll.sum().cpu(); outM[i, a_, b] = (~inside).sum().cpu()
        if i % 100 == 0:
            print(i, R, f"{time.time() - t0:.0f}s", flush=True)
    meta = [{"endpoint": r["endpoint"], "id": r["id"], "ntok": len(x)} for r, _, x in enc]
    torch.save({"meta": meta, "TS": TS, "PS": PS, "KS": KS, "MPS": MPS, "llP": llP, "outP": outP, "llK": llK,
                "outK": outK, "llM": llM, "outM": outM}, common.DATA / f"dec_{a.tag}.pt")
    # summary
    ep_of = np.array([m["endpoint"] for m in meta])
    print(f"== {a.tag}: best decoding per endpoint (ll in nats; d = gain over best plain temperature)")
    for e in sorted(set(ep_of)):
        m = torch.tensor(ep_of == e)
        n = sum(meta[j]["ntok"] for j in np.where(ep_of == e)[0])
        P_ = llP[m].sum(0); K_ = llK[m].sum(0); M_ = llM[m].sum(0)
        jT = int(P_[:, -1].argmax()); base = float(P_[jT, -1])
        iP = np.unravel_index(int(P_.argmax()), P_.shape); iK = np.unravel_index(int(K_.argmax()), K_.shape)
        iM = np.unravel_index(int(M_.argmax()), M_.shape)
        print(f"  {e}: T-only T={TS[jT]:.2f} ll={base:9.1f} | top-p T={TS[iP[0]]:.2f} P={PS[iP[1]]:.2f} "
              f"d={float(P_[iP]) - base:+7.1f} out={int(outP[m].sum(0)[iP])} | top-k T={TS[iK[0]]:.2f} "
              f"K={KS[iK[1]]} d={float(K_[iK]) - base:+7.1f} out={int(outK[m].sum(0)[iK])} | min-p "
              f"T={TS[iM[0]]:.2f} mp={MPS[iM[1]]} d={float(M_[iM]) - base:+7.1f} out={int(outM[m].sum(0)[iM])} n={n}")


if __name__ == "__main__":
    main()
