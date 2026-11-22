"""Score a data file under several candidate models and print mean token logprob per endpoint x model.
Usage: python family_table.py DATA_NAME model1 model2 ..."""
import gc
import pickle
import sys

import numpy as np
import pandas as pd
import torch

import common
from score import load_model, score_rows

data = sys.argv[1]
models = sys.argv[2:]
rows = common.load(data)
table = {}
for mid in models:
    model, tok = load_model(mid)
    res = score_rows(model, tok, rows)
    tag = mid.split("/")[-1]
    with open(common.DATA / f"scores_{data}__{tag}.pkl", "wb") as f:
        pickle.dump(res, f)
    by = {}
    for r in res:
        by.setdefault(r["endpoint"], []).append(r["lp"])
    table[tag] = {ep: float(np.concatenate(v).mean()) for ep, v in by.items()}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("done", tag, flush=True)
df = pd.DataFrame(table)
df["best"] = df.idxmax(axis=1)
pd.set_option("display.width", 250)
print(df.round(3).to_string())
df.to_csv(common.DATA / f"family_{data}.csv")
