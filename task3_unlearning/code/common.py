"""Shared helpers for the unlearning task. Imports the official helpers from ../task_template.py (read-only)."""
import importlib.util
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
TASK_DIR = Path(os.environ.get("TASK_DIR", HERE.parent)).resolve()
WORK = Path(os.environ.get("UNL_WORK", HERE)).resolve()  # where artifacts/outputs are written
ART = WORK / "artifacts"
OUT = WORK / "output"
ANCHORS = Path(os.environ.get("UNL_ANCHORS", HERE / "anchors" / "anchors.npz"))
ART.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


def _load_template():
    spec = importlib.util.spec_from_file_location("task_template", TASK_DIR / "task_template.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tt = _load_template()
DEVICE = tt.DEVICE
MODEL_PATH = Path(os.environ.get("MODEL_PATH", TASK_DIR / "model.pt"))
FORGET_PATH = Path(os.environ.get("FORGET_PATH", TASK_DIR / "forget_images.npz"))
NUM_CLASSES = tt.NUM_CLASSES


def seed_all(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_orig():
    return tt.load_model(MODEL_PATH)


def load_images_gpu():
    return tt.load_forget_images(FORGET_PATH).to(DEVICE)  # uint8 [N,224,224,3]


def save_state_dict(sd_or_model, path):
    sd = sd_or_model.state_dict() if hasattr(sd_or_model, "state_dict") else sd_or_model
    sd = {k: v.detach().cpu().float().contiguous() for k, v in sd.items()}
    torch.save(sd, path)


def fold_split(n, frac_train=0.8, seed=1234):
    g = np.random.default_rng(seed)
    perm = g.permutation(n)
    k = int(round(n * frac_train))
    return np.sort(perm[:k]), np.sort(perm[k:])


def kfold_split(n, k, K=5, seed=1234):
    """fold k of K: (train = other folds, holdout = fold k)."""
    perm = np.random.default_rng(seed).permutation(n)
    ho = np.sort(perm[k::K])
    tr = np.sort(np.setdiff1d(np.arange(n), ho))
    return tr, ho


# ---------------------------------------------------------------- pixel-space GPU augmentations (x in [0,1], B3HW)
def u8_to01(u8):
    return u8.permute(0, 3, 1, 2).float() / 255.0


def norm(x01):
    return (x01 - 0.5) / 0.5


def _u(B, lo, hi, gen, dev):
    return torch.rand(B, device=dev, generator=gen) * (hi - lo) + lo


def _affine(x, gen, scale=(0.6, 1.0), ratio=(0.8, 1.25), flip=True, rot=0.0):
    B, dev = x.shape[0], x.device
    area = _u(B, *scale, gen, dev)
    logr = _u(B, math.log(ratio[0]), math.log(ratio[1]), gen, dev)
    r = torch.exp(logr)
    sx = torch.sqrt(area * r).clamp(max=1.0)
    sy = torch.sqrt(area / r).clamp(max=1.0)
    tx = _u(B, -1, 1, gen, dev) * (1 - sx)
    ty = _u(B, -1, 1, gen, dev) * (1 - sy)
    ang = _u(B, -rot, rot, gen, dev) * math.pi / 180
    f = torch.where(_u(B, 0, 1, gen, dev) < 0.5, -1.0, 1.0) if flip else torch.ones(B, device=dev)
    c, s = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([torch.stack([sx * c * f, -sy * s, tx], 1),
                         torch.stack([sx * s * f, sy * c, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def _color(x, gen, b=0.2, c=0.2, s=0.2):
    B, dev = x.shape[0], x.device
    if b:
        x = x * _u(B, 1 - b, 1 + b, gen, dev).view(B, 1, 1, 1)
    if c:
        m = x.mean((1, 2, 3), keepdim=True)
        x = (x - m) * _u(B, 1 - c, 1 + c, gen, dev).view(B, 1, 1, 1) + m
    if s:
        g = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
        x = (x - g) * _u(B, 1 - s, 1 + s, gen, dev).view(B, 1, 1, 1) + g
    return x.clamp(0, 1)


def _blur(x, sigma):
    k = max(3, int(2 * round(3 * sigma) + 1))
    t = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    w = torch.exp(-t ** 2 / (2 * sigma ** 2))
    w = w / w.sum()
    p = k // 2
    x = F.conv2d(F.pad(x, (p, p, 0, 0), mode="reflect"), w.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    x = F.conv2d(F.pad(x, (0, 0, p, p), mode="reflect"), w.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)
    return x


def _maybe_blur(x, gen, p, smin, smax):
    if p <= 0:
        return x
    sigma = float(_u(1, smin, smax, gen, x.device))
    m = (_u(x.shape[0], 0, 1, gen, x.device) < p).view(-1, 1, 1, 1)
    return torch.where(m, _blur(x, sigma), x)


def augment(x01, kind, gen):
    """kind: clean | mild (training-like) | strong (difficulty proxy) | heldout (eval-only family, never trained on)"""
    if kind == "clean":
        return x01
    if kind == "mild":
        x = _affine(x01, gen, scale=(0.6, 1.0), flip=True, rot=10)
        x = _color(x, gen, 0.2, 0.2, 0.2)
        x = _maybe_blur(x, gen, 0.2, 0.3, 1.0)
    elif kind == "strong":
        x = _affine(x01, gen, scale=(0.1, 0.4), ratio=(0.75, 1.333), flip=True, rot=0)
        x = _color(x, gen, 0.4, 0.4, 0.4)
        x = _maybe_blur(x, gen, 0.5, 0.5, 2.0)
    elif kind == "heldout":
        x = _affine(x01, gen, scale=(0.75, 0.9), ratio=(0.9, 1.1), flip=False, rot=15)
        x = _color(x, gen, 0.0, 0.3, 0.0)
        x = (x + 0.03 * torch.randn(x.shape, device=x.device, generator=gen)).clamp(0, 1)
    else:
        raise ValueError(kind)
    return torch.round(x * 255) / 255  # server images are uint8


def make_gen(seed):
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)
    return g


@torch.no_grad()
def logits_of(model, imgs_u8, idx=None, kind="clean", seed=0, bs=128, fp16=True):
    """Logits [len(idx),100] for images (optionally augmented with a deterministic seed), server-like fp16 autocast."""
    model.eval()
    idx = np.arange(len(imgs_u8)) if idx is None else np.asarray(idx)
    gen = make_gen(seed)
    out = []
    for i in range(0, len(idx), bs):
        x = augment(u8_to01(imgs_u8[torch.as_tensor(idx[i:i + bs], device=imgs_u8.device)]), kind, gen)
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and DEVICE == "cuda"):
            out.append(model(norm(x)).float())
    return torch.cat(out)


def ce_vec(logits, y):
    return F.cross_entropy(logits.float(), y, reduction="none")


def kl_vec(p_logits, q_logits, T=2.0):
    """KL(softmax(p/T) || softmax(q/T)) per sample."""
    lp = F.log_softmax(p_logits.float() / T, -1)
    lq = F.log_softmax(q_logits.float() / T, -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def dump_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=float))
