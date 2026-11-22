"""Detect a KGW-style watermark (HF transformers WatermarkLogitsProcessor defaults) in endpoint responses.
Green-token fraction should be ~gamma (0.25) without watermark, clearly higher with it.

Usage: python wm_check.py TOKENIZER_ID DATA_NAME [DATA_NAME ...] [--eps ep-01,ep-02]
"""
import sys

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.generation.logits_process import WatermarkLogitsProcessor

import common
from score import build

args = sys.argv[1:]
keep = None
if "--eps" in args:
    i = args.index("--eps"); keep = set(args[i + 1].split(",")); del args[i:i + 2]
mid, datas = args[0], args[1:]
tok = AutoTokenizer.from_pretrained(mid)
V = len(tok)
configs = {
    "hf_default(g.25,lefthash)": dict(greenlist_ratio=0.25, bias=2.0, hashing_key=15485863, seeding_scheme="lefthash", context_width=1),
}
rows = [r for d in datas for r in common.load(d) if keep is None or r["endpoint"] in keep]
for cname, cfg in configs.items():
    # vocab size must match the model's embedding size used at generation time; try tokenizer length
    proc = WatermarkLogitsProcessor(vocab_size=V, device="cpu", **cfg)
    by = {}
    cache = {}

    def green_set(key_ids):
        k = tuple(key_ids)
        if k not in cache:
            cache[k] = set(proc._get_greenlist_ids(torch.tensor(key_ids)).tolist())
        return cache[k]

    for r in rows:
        ctx = tok(build(tok, r["prompt"], None), add_special_tokens=False)["input_ids"]
        resp = tok(r["response"], add_special_tokens=False)["input_ids"]
        ids = ctx + resp
        g = n = 0
        for t in range(len(ctx), len(ids)):
            if cfg["seeding_scheme"] == "lefthash":
                green = green_set(ids[t - cfg["context_width"]:t])
            else:
                green = green_set(ids[t - cfg["context_width"]:t + 1])
            g += int(ids[t] in green)
            n += 1
        a = by.setdefault(r["endpoint"], [0, 0]); a[0] += g; a[1] += n
    print(f"== {cname}")
    for ep in sorted(by):
        g, n = by[ep]
        frac = g / max(n, 1)
        z = (g - 0.25 * n) / np.sqrt(n * 0.25 * 0.75 + 1e-9)
        print(f"  {ep}: green={frac:.3f}  z={z:+.1f}  n={n}")
