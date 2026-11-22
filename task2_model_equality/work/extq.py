"""Load the weights of an official pre-quantised checkpoint (AWQ gemm, GPTQ sym, compressed-tensors int8/fp8) into a
full-precision copy of the base model, by de-quantising every linear layer in pure PyTorch.
Activation quantisation of the w8a8 schemes is simulated with dynamic per-token forward pre-hooks."""
import glob
import json
import os

import torch
from safetensors import safe_open

AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

EXTERNAL = {
    "Qwen/Qwen2.5-1.5B-Instruct": {"awq": "Qwen/Qwen2.5-1.5B-Instruct-AWQ", "gptq4": "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",
                                   "gptq8": "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8"},
    "Qwen/Qwen2.5-3B-Instruct": {"awq": "Qwen/Qwen2.5-3B-Instruct-AWQ", "gptq4": "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
                                 "gptq8": "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8"},
    "unsloth/Llama-3.2-1B-Instruct": {"ctint8": "RedHatAI/Llama-3.2-1B-Instruct-quantized.w8a8",
                                      "ctfp8": "RedHatAI/Llama-3.2-1B-Instruct-FP8-dynamic"},
}


def snapshot(repo):
    return glob.glob(os.path.join(os.environ["HF_HOME"], "hub", "models--" + repo.replace("/", "--"), "snapshots", "*"))[0]


def tensors(d):
    out = {}
    for f in glob.glob(d + "/*.safetensors"):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                out[k] = s.get_tensor(k)
    return out


def unpack_cols(q, bits):          # [r, c] int32 -> [r, c * 32/bits] (value j of word at shift bits*j)
    shifts = torch.arange(0, 32, bits, device=q.device)
    return ((q[:, :, None] >> shifts[None, None, :]) & ((1 << bits) - 1)).reshape(q.shape[0], -1)


def unpack_rows(q, bits):          # [r, c] int32 -> [r * 32/bits, c]
    shifts = torch.arange(0, 32, bits, device=q.device)
    return ((q[:, None, :] >> shifts[None, :, None]) & ((1 << bits) - 1)).reshape(-1, q.shape[1])


def deq_awq(t, p, bits=4, group=128):
    iw = unpack_cols(t[p + ".qweight"], bits)                # [in, out]
    iz = unpack_cols(t[p + ".qzeros"], bits)                 # [in/group, out]
    order = torch.arange(iw.shape[1]).view(-1, 32 // bits)[:, AWQ_REVERSE_ORDER].reshape(-1)
    iw, iz = iw[:, order], iz[:, order]
    sc = t[p + ".scales"].float()                            # [in/group, out]
    W = (iw.float() - iz.float().repeat_interleave(group, 0)) * sc.repeat_interleave(group, 0)
    return W.T.contiguous()                                  # [out, in]


def deq_gptq(t, p, bits, group=128):
    iw = unpack_rows(t[p + ".qweight"], bits)                # [in, out]
    iz = unpack_cols(t[p + ".qzeros"], bits) + 1             # [groups, out]  (auto-gptq stores zero - 1)
    sc = t[p + ".scales"].float()
    g = t[p + ".g_idx"].long()
    W = (iw.float() - iz.float()[g]) * sc[g]
    return W.T.contiguous()


def deq_ct(t, p):
    w = t[p + ".weight"]
    return w.float() * t[p + ".weight_scale"].float()


def act_hook(kind):
    def hook(mod, args):
        a = args[0]
        af = a.float()
        if kind == "int8":
            s = af.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 127.0
            q = torch.round(af / s).clamp(-127, 127) * s
        else:
            s = af.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
            q = (af / s).to(torch.float8_e4m3fn).float() * s
        return (q.to(a.dtype),)
    return hook


@torch.no_grad()
def apply_external(model, repo, scheme):
    d = snapshot(repo)
    t = tensors(d)
    hooks, errs = [], []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or "lm_head" in name:
            continue
        p = name
        if scheme == "awq":
            W = deq_awq(t, p)
        elif scheme.startswith("gptq"):
            W = deq_gptq(t, p, int(scheme[4:]))
        else:
            W = deq_ct(t, p)
        W = W.to(mod.weight.device)
        errs.append(((W - mod.weight.float()).norm() / mod.weight.float().norm()).item())
        mod.weight.data = W.to(mod.weight.dtype)
        if p + ".bias" in t and mod.bias is not None:
            mod.bias.data = t[p + ".bias"].to(mod.bias.device, mod.bias.dtype)
        if scheme == "ctint8":
            hooks.append(mod.register_forward_pre_hook(act_hook("int8")))
        elif scheme == "ctfp8":
            hooks.append(mod.register_forward_pre_hook(act_hook("fp8")))
    # every other tensor of the checkpoint (norms with folded AWQ / SmoothQuant scales, embeddings, biases)
    params = dict(model.named_parameters())
    copied = 0
    for k, v in t.items():
        if k in params and not k.endswith(("qweight", "qzeros", "scales", "g_idx", "weight_scale")):
            lin = k.rsplit(".", 1)[0]
            if k.endswith(".weight") and isinstance(model.get_submodule(lin), torch.nn.Linear) and "lm_head" not in k:
                continue
            if params[k].shape == v.shape:
                params[k].data = v.to(params[k].device, params[k].dtype)
                copied += 1
    print(f"  copied {copied} other tensors from the checkpoint", flush=True)
    print(f"  external {scheme} ({repo}): {len(errs)} linears, rel.err median {sorted(errs)[len(errs) // 2]:.4f} "
          f"max {max(errs):.4f}", flush=True)
    return hooks
