"""Local calibration on near-deterministic quiz prompts.
1. Sample N answers from the local base model (and its fp8 copy) at T=0.7 with the default chat template.
2. Log-likelihood (T=0.7 and T=1.0) of every endpoint quiz answer and of every local sample, under base and fp8.
Output: data/qcal_<TAG>.json

Usage: python qcal.py TAG MODEL_ID --eps a,b,c [--n 200]
"""
import argparse
import copy
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import common
from llr import quantize_
from score import build

DEV = "cuda"
SOURCES = [("ep_a", "quiz"), ("quizA_ep", "quizA"), ("quiz2_ep", "quiz2")]


@torch.no_grad()
def sample(model, tok, prompt, n, T=0.7, bs=50):
    ids = tok(build(tok, prompt, None), return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
    out = []
    for i in range(0, n, bs):
        g = model.generate(ids.repeat(min(bs, n - i), 1), do_sample=True, temperature=T, top_p=1.0, top_k=0,
                           max_new_tokens=256, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        out += tok.batch_decode(g[:, ids.shape[1]:], skip_special_tokens=True)
    return out


@torch.no_grad()
def loglik(model, tok, prompt, resp, Ts=(0.7, 1.0)):
    c = tok(build(tok, prompt, None), add_special_tokens=False)["input_ids"]
    x = tok(resp, add_special_tokens=False)["input_ids"]
    if len(x) < 250:
        x = x + [tok.eos_token_id]
    ids = torch.tensor([c + x], device=DEV)
    lg = model(ids).logits[0, len(c) - 1:-1].float()
    xt = ids[0, len(c):]
    return {str(T): float(torch.log_softmax(lg / T, -1).gather(1, xt[:, None]).sum()) for T in Ts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("model")
    ap.add_argument("--eps", required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    keep = set(a.eps.split(","))
    tok = AutoTokenizer.from_pretrained(a.model)
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map=DEV).eval()
    fp8 = copy.deepcopy(base).float()
    quantize_(fp8, "fp8_tensor")
    fp8 = fp8.to(torch.bfloat16)
    prompts, eprs = {}, []
    for data, pid in SOURCES:
        for r in common.load(data):
            if r["id"] == pid and r["endpoint"] in keep:
                prompts[pid] = r["prompt"]
                eprs.append({"endpoint": r["endpoint"], "pid": pid, "response": r["response"]})
    res = {"model": a.model, "endpoint_answers": [], "samples": {}}
    for pid, prompt in prompts.items():
        for name, m, T in (("base", base, 0.7), ("fp8", fp8, 0.7), ("base_T1", base, 1.0)):
            torch.manual_seed(0)
            S = sample(m, tok, prompt, a.n, T=T)
            res["samples"][f"{pid}|{name}"] = [{"response": s, "ll_base": loglik(base, tok, prompt, s),
                                                "ll_fp8": loglik(fp8, tok, prompt, s)} for s in S]
            print(pid, name, "sampled", len(S), flush=True)
    for r in eprs:
        p = prompts[r["pid"]]
        res["endpoint_answers"].append(r | {"ll_base": loglik(base, tok, p, r["response"]),
                                            "ll_fp8": loglik(fp8, tok, p, r["response"])})
    json.dump(res, open(common.DATA / f"qcal_{a.tag}.json", "w"))
    print("saved", flush=True)


if __name__ == "__main__":
    main()
