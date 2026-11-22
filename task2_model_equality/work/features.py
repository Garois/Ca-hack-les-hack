"""Per-endpoint features from local-model scores, binned by the local model's confidence.
Usage: python features.py DATA_NAME MODEL_TAG [DATA_NAME2 ...]  (several data names are pooled)"""
import pickle
import sys

import numpy as np
import pandas as pd

import common

BINS = [(0.99, 1.01), (0.9, 0.99), (0.7, 0.9), (0.5, 0.7), (0.0, 0.5)]


def load_scores(datas, tag):
    res = []
    for d in datas:
        with open(common.DATA / f"scores_{d}__{tag}.pkl", "rb") as f:
            res += pickle.load(f)
    return res


def endpoint_tokens(res, skip_first=0):
    by = {}
    for r in res:
        s = slice(skip_first, None)
        by.setdefault(r["endpoint"], []).append({k: r[k][s] for k in ("lp", "rank", "ent", "ptop")})
    return {ep: {k: np.concatenate([x[k] for x in v]) for k in v[0]} for ep, v in by.items()}


def feats(tok):
    pmax = np.exp(tok["ptop"])
    out = {"n": len(pmax), "mean_lp": tok["lp"].mean(), "top1": np.mean(tok["rank"] == 0)}
    for lo, hi in BINS:
        m = (pmax >= lo) & (pmax < hi)
        out[f"dis[{lo}]"] = np.mean(tok["rank"][m] > 0) if m.any() else np.nan     # disagreement with local argmax
        out[f"n[{lo}]"] = int(m.sum())
    # expected disagreement under the local model itself at T=1 in the top bin, for reference
    m = pmax >= 0.99
    out["exp_dis_T1[.99]"] = np.mean(1 - pmax[m]) if m.any() else np.nan
    out["far"] = np.mean(tok["rank"] >= 10)
    return out


if __name__ == "__main__":
    tag = sys.argv[2]
    datas = [sys.argv[1]] + sys.argv[3:]
    tok = endpoint_tokens(load_scores(datas, tag))
    df = pd.DataFrame({ep: feats(t) for ep, t in tok.items()}).T.sort_index()
    pd.set_option("display.width", 250)
    print(df.round(4).to_string())
