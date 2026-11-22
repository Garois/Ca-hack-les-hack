"""Score endpoint responses under a local model: per-token logprob, rank, entropy of the local next-token distribution.

Usage: python score.py MODEL_ID DATA_NAME [OUT_TAG] [--system "text"|--nosys]
Writes data/scores_<DATA_NAME>__<OUT_TAG>.pkl : list of dicts (endpoint,id,repeat,lp[],rank[],ent[],ptop[])
"""
import argparse
import pickle

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import common


def build(tok, prompt, system):
    msgs = ([{"role": "system", "content": system}] if system is not None else []) + [{"role": "user", "content": prompt}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def score_rows(model, tok, rows, system=None, device="cuda", bs=8):
    out = []
    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]
        for r in batch:
            ctx = build(tok, r["prompt"], system)
            ids_ctx = tok(ctx, add_special_tokens=False)["input_ids"]
            ids_resp = tok(r["response"], add_special_tokens=False)["input_ids"]
            if not ids_resp:
                continue
            ids = torch.tensor([ids_ctx + ids_resp], device=device)
            logits = model(ids).logits[0, len(ids_ctx) - 1:-1].float()
            logp = torch.log_softmax(logits, -1)
            tgt = ids[0, len(ids_ctx):]
            lp = logp.gather(1, tgt[:, None])[:, 0]
            rank = (logp > lp[:, None]).sum(1)
            ent = -(logp.exp() * logp).sum(1)
            ptop = logp.max(1).values
            out.append({k: r[k] for k in ("endpoint", "id", "repeat")} | {
                "lp": lp.cpu().numpy(), "rank": rank.cpu().numpy(), "ent": ent.cpu().numpy(), "ptop": ptop.cpu().numpy()})
    return out


def load_model(mid, dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=dtype, device_map="cuda").eval()
    return model, tok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("data")
    ap.add_argument("tag", nargs="?", default=None)
    ap.add_argument("--system", default=None)
    ap.add_argument("--nosys", action="store_true", help="force an empty system turn (disables default system prompt)")
    ap.add_argument("--fp32", action="store_true")
    a = ap.parse_args()
    model, tok = load_model(a.model, torch.float32 if a.fp32 else torch.bfloat16)
    system = "" if a.nosys else a.system
    rows = common.load(a.data)
    res = score_rows(model, tok, rows, system)
    tag = a.tag or a.model.split("/")[-1]
    with open(common.DATA / f"scores_{a.data}__{tag}.pkl", "wb") as f:
        pickle.dump(res, f)
    # quick summary: mean logprob per endpoint
    by = {}
    for r in res:
        by.setdefault(r["endpoint"], []).append(r)
    print(f"model={a.model} data={a.data} n={len(res)}")
    for ep in sorted(by):
        lp = np.concatenate([r["lp"] for r in by[ep]])
        rk = np.concatenate([r["rank"] for r in by[ep]])
        print(f"  {ep}: mean_lp={lp.mean():.3f}  top1={np.mean(rk == 0):.3f}  top5={np.mean(rk < 5):.3f}  ntok={len(lp)}")
