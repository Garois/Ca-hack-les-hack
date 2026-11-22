"""Calibration statistics of endpoint tokens under the local base model at the endpoint's own fitted temperature.

For each endpoint: T fitted by MLE (fine grid, no top-p). Then per token, with p_T = softmax(logits/T):
  - tail counts: observed #tokens with p_T(x) < tau  vs expected sum_t P_{X~p_T}(p_T(X) < tau)
  - randomized PIT u (tokens sorted by p_T descending): uniform under H0. 10-bin histogram -> chi2
  - argmax agreement observed vs expected sum p_T,max
Saves per-response arrays to data/tail_<data>__<fam>.pkl so that statistics can be recomputed / bootstrapped.

Usage: python tailstat.py --fam Q15 DATA [DATA ...]     or    python tailstat.py --model MODEL_ID DATA [DATA ...]
"""
import pickle
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import common
from score import build

TAUS = [1e-2, 1e-3, 1e-4]
TGRID = np.round(np.arange(0.3, 1.6, 0.025), 3)


@torch.no_grad()
def logits_for(model, tok, r):
    ids_ctx = tok(build(tok, r["prompt"], None), add_special_tokens=False)["input_ids"]
    ids_resp = tok(r["response"], add_special_tokens=False)["input_ids"]
    if len(ids_resp) < 2:
        return None, None
    ids = torch.tensor([ids_ctx + ids_resp], device="cuda")
    return model(ids).logits[0, len(ids_ctx) - 1:-1].float(), ids[0, len(ids_ctx):]


@torch.no_grad()
def fit_T(cache):
    """cache: list of (logits, tgt). MLE over TGRID."""
    best, bestT = -1e18, None
    for T in TGRID:
        s = 0.0
        for lg, tg in cache:
            s += torch.log_softmax(lg / float(T), -1).gather(1, tg[:, None]).sum().item()
        if s > best:
            best, bestT = s, float(T)
    return bestT


@torch.no_grad()
def stats_for(lg, tg, T, gen):
    lp = torch.log_softmax(lg / T, -1)
    p = lp.exp()
    pt = p.gather(1, tg[:, None])[:, 0]
    out = {"lp": torch.log(pt).cpu().numpy()}
    for tau in TAUS:
        out[f"obs{tau}"] = (pt < tau).float().cpu().numpy()
        out[f"exp{tau}"] = (p * (p < tau)).sum(1).cpu().numpy()
    before = (p * (p > pt[:, None])).sum(1)
    u = before + torch.rand(len(pt), generator=gen, device=p.device) * pt
    out["u"] = u.cpu().numpy()
    pmax = p.max(1).values
    out["agree_obs"] = (pt >= pmax).float().cpu().numpy()
    out["agree_exp"] = pmax.cpu().numpy()
    out["pmax"] = pmax.cpu().numpy()
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0] == "--fam":
        from fams import FAMILIES
        tag = args[1]; mid, keep = FAMILIES[tag]; datas = args[2:]
    else:
        mid = args[1]; tag = mid.split("/")[-1]; keep = None; datas = args[2:]
    rows = [r for d in datas for r in common.load(d) if keep is None or r["endpoint"] in keep]
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32, device_map="cuda").eval()
    gen = torch.Generator(device="cuda").manual_seed(0)
    by = {}
    for r in rows:
        by.setdefault(r["endpoint"], []).append(r)
    result = {}
    for ep in sorted(by):
        cache, meta = [], []
        for r in by[ep]:
            lg, tg = logits_for(model, tok, r)
            if lg is not None:
                cache.append((lg, tg)); meta.append(r["id"])
        T = fit_T(cache)
        per = [stats_for(lg, tg, T, gen) | {"id": i} for (lg, tg), i in zip(cache, meta)]
        result[ep] = {"T": T, "per": per}
        cat = {k: np.concatenate([x[k] for x in per]) for k in per[0] if k != "id"}
        n = len(cat["u"])
        h = np.histogram(cat["u"], bins=10, range=(0, 1))[0]
        chi2 = ((h - n / 10) ** 2 / (n / 10)).sum()
        tails = "  ".join(f"r{tau:g}={cat[f'obs{tau}'].sum() / cat[f'exp{tau}'].sum():.2f}" for tau in TAUS)
        ag = cat["agree_obs"].sum() / cat["agree_exp"].sum()
        print(f"  {ep}: T={T:.3f} n={n:5d}  {tails}  agree_ratio={ag:.3f}  pit_chi2={chi2:7.1f}  "
              f"u>.99:{np.mean(cat['u'] > .99) * 100:.2f}%", flush=True)
    with open(common.DATA / f"tail_{'+'.join(datas)}__{tag}.pkl", "wb") as f:
        pickle.dump(result, f)
