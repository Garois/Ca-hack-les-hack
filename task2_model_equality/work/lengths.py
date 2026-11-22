"""Response length / natural-stop statistics per endpoint (tokens of the family base tokenizer).
Usage: python lengths.py DATA [DATA ...]"""
import sys

import numpy as np
from transformers import AutoTokenizer

import common
from fams import FAMILIES

rows = [r for d in sys.argv[1:] for r in common.load(d)]
for fam, (m, eps) in FAMILIES.items():
    tok = AutoTokenizer.from_pretrained(m)
    print(f"== {fam}")
    for e in eps:
        L = np.array([len(tok(r["response"], add_special_tokens=False)["input_ids"]) for r in rows if r["endpoint"] == e])
        stop = L < 250
        print(f"  {e}: n={len(L)} stop%={100 * stop.mean():5.1f}  mean_len_if_stop={L[stop].mean() if stop.any() else 0:6.1f}  "
              f"median={np.median(L):.0f}")
