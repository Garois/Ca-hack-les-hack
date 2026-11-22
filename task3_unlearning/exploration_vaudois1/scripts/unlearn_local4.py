#!/usr/bin/env python3
"""Unlearning « local » (vaudois1) -- cf. PLAN.md §5, SCAR (ECCV'24), LayerMem (NeurIPS'24, CISPA).

  * distillation (T=2, + terme T=1) vers des cibles souples q(x) sur les images forget et leurs vues alterees
    (cible des vues = logits orig(vue) - delta*e_y, comme R13 ; ou la meme q avec --view_target same),
  * en ne modifiant qu'une petite partie du reseau (--train, defaut : MLP des blocs 8-11) avec un masque de
    saillance optionnel facon SalUn (--mask_frac),
  * avec une distillation forte vers l'original sur des ancres (Imagenette) = retain de substitution
    (--w_anc, KL a T=2 comme la metrique), + ancrage des features pre-logits (--w_feat) + L2-SP (--l2sp).

Metriques locales (par alpha d'interpolation orig -> entraine) :
  forget : CE moyenne/ecart-type sur images propres et 3 familles d'alterations (mild, heldout, jpeg q=70 --
           jpeg n'est jamais vu a l'entrainement), KL vers la cible, KL vers orig
  retain : KL(orig||sub) a T=2 et accord top-1 sur 1500 ancres val jamais vues ; retain_est = accord/(1+KL)
  holdout (--holdout f) : KL vs orig sur une part du forget NON entrainee (debordement sur la classe)
Ecrit runs/<tag>/alpha_<a>.pt (+ metrics.json) et une ligne par alpha dans RESULTS_LOCAL.md.
--eval_ckpts a.pt,b.pt : mesure seulement des checkpoints existants (pour relier metriques locales et scores publics).

Depuis unlearning/vaudois1 (GPU) :
  ../.venv/bin/python -I scripts/unlearn_local.py --targets ../lemasnedechermont1/artifacts/targets_s_mu36_b15.npz --tag L1
"""
import argparse
import copy
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]  # unlearning/vaudois1
TASK = ROOT.parent  # unlearning/
SEEDS = {"mild": 101, "heldout": 202, "medium": 303}


# ------------------------------------------------------------------------------------------------ model / io
def build():
    return timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100, init_values=1e-4)


def load_sd(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def u8_to01(u8):
    return u8.permute(0, 3, 1, 2).float() / 255.0


def to_u8(x01):
    return (x01.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def norm(x01):
    return (x01 - 0.5) / 0.5


def make_gen(dev, seed):
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    return g


# ------------------------------------------------------------------------------------------------ augmentations
def _u(B, lo, hi, gen, dev):
    return torch.rand(B, device=dev, generator=gen) * (hi - lo) + lo


def _affine(x, gen, scale=(0.6, 1.0), ratio=(0.8, 1.25), flip=True, rot=0.0):
    B, dev = x.shape[0], x.device
    area = _u(B, *scale, gen, dev)
    r = torch.exp(_u(B, math.log(ratio[0]), math.log(ratio[1]), gen, dev))
    sx = torch.sqrt(area * r).clamp(max=1.0)
    sy = torch.sqrt(area / r).clamp(max=1.0)
    tx = _u(B, -1, 1, gen, dev) * (1 - sx)
    ty = _u(B, -1, 1, gen, dev) * (1 - sy)
    ang = _u(B, -rot, rot, gen, dev) * math.pi / 180
    f = torch.where(_u(B, 0, 1, gen, dev) < 0.5, -1.0, 1.0) if flip else torch.ones(B, device=dev)
    c, s = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([torch.stack([sx * c * f, -sy * s, tx], 1), torch.stack([sx * s * f, sy * c, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def _color(x, gen, b=0.2, c=0.2, s=0.2):
    B, dev = x.shape[0], x.device
    if b:
        x = x * _u(B, 1 - b, 1 + b, gen, dev).view(B, 1, 1, 1)
    if c:
        m = x.mean((1, 2, 3), keepdim=True)
        x = (x - m) * _u(B, 1 - c, 1 + c, gen, dev).view(B, 1, 1, 1) + m
    if s:
        g = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        x = (x - g) * _u(B, 1 - s, 1 + s, gen, dev).view(B, 1, 1, 1) + g
    return x.clamp(0, 1)


def _blur(x, sigma):
    k = max(3, int(2 * round(3 * sigma) + 1))
    t = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    w = torch.exp(-t ** 2 / (2 * sigma ** 2))
    w = w / w.sum()
    p = k // 2
    x = F.conv2d(F.pad(x, (p, p, 0, 0), mode="reflect"), w.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(F.pad(x, (0, 0, p, p), mode="reflect"), w.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)


def _maybe_blur(x, gen, p, smin, smax):
    sigma = float(_u(1, smin, smax, gen, x.device))
    m = (_u(x.shape[0], 0, 1, gen, x.device) < p).view(-1, 1, 1, 1)
    return torch.where(m, _blur(x, sigma), x)


def augment(x01, kind, gen):
    """clean | mild (proche de l'entrainement) | heldout (rotation+bruit) | strong."""
    if kind == "clean":
        return x01
    if kind == "mild":
        x = _affine(x01, gen, scale=(0.6, 1.0), flip=True, rot=10)
        x = _color(x, gen, 0.2, 0.2, 0.2)
        x = _maybe_blur(x, gen, 0.2, 0.3, 1.0)
    elif kind == "heldout":
        x = _affine(x01, gen, scale=(0.75, 0.9), ratio=(0.9, 1.1), flip=False, rot=15)
        x = _color(x, gen, 0.0, 0.3, 0.0)
        x = (x + 0.03 * torch.randn(x.shape, device=x.device, generator=gen)).clamp(0, 1)
    elif kind == "medium":
        x = _affine(x01, gen, scale=(0.2, 0.5), ratio=(0.75, 1.333), flip=True, rot=10)
        x = _color(x, gen, 0.3, 0.3, 0.3)
        x = _maybe_blur(x, gen, 0.3, 0.5, 1.5)
    elif kind == "strong":
        x = _affine(x01, gen, scale=(0.1, 0.4), ratio=(0.75, 1.333), flip=True, rot=0)
        x = _color(x, gen, 0.4, 0.4, 0.4)
        x = _maybe_blur(x, gen, 0.5, 0.5, 2.0)
    else:
        raise ValueError(kind)
    return torch.round(x * 255) / 255


def jpeg_u8(u8_nhwc, quality=70):
    import torchvision.io as tio
    out = []
    for im in u8_nhwc.cpu():
        enc = tio.encode_jpeg(im.permute(2, 0, 1).contiguous(), quality=quality)
        out.append(tio.decode_jpeg(enc).permute(1, 2, 0))
    return torch.stack(out).to(u8_nhwc.device)


@torch.no_grad()
def eval_views(imgs, dev):
    """Vues d'evaluation fixes (uint8), calculees une fois."""
    v = {"clean": imgs}
    for kind in ("mild", "heldout"):
        gen = make_gen(dev, SEEDS[kind])
        v[kind] = torch.cat([to_u8(augment(u8_to01(imgs[i:i + 256]), kind, gen)) for i in range(0, len(imgs), 256)])
    v["jpeg"] = torch.cat([jpeg_u8(imgs[i:i + 256]) for i in range(0, len(imgs), 256)])
    gen = make_gen(dev, SEEDS["medium"])
    v["medium"] = torch.cat([to_u8(augment(u8_to01(imgs[i:i + 256]), "medium", gen)) for i in range(0, len(imgs), 256)])
    return v


# ------------------------------------------------------------------------------------------------ forward / losses
def fwd(m, x01, feat=False):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x01.is_cuda):
        f = m.forward_features(norm(x01))
        pre = m.forward_head(f, pre_logits=True)
        z = m.head(pre)
    return (z.float(), pre.float()) if feat else z.float()


@torch.no_grad()
def logits_of(m, u8, bs=256):
    """Comme le serveur : eval, autocast fp16."""
    m.eval()
    out = []
    for i in range(0, len(u8), bs):
        with torch.autocast("cuda", dtype=torch.float16, enabled=u8.is_cuda):
            out.append(m(norm(u8_to01(u8[i:i + bs]))).float())
    return torch.cat(out)


def kd(t_logits, s_logits, T, w_t1=0.0):
    lt, ls = F.log_softmax(t_logits.float() / T, -1), F.log_softmax(s_logits.float() / T, -1)
    loss = (lt.exp() * (lt - ls)).sum(-1).mean() * T * T
    if w_t1 > 0:
        lt1, ls1 = F.log_softmax(t_logits.float(), -1), F.log_softmax(s_logits.float(), -1)
        loss = loss + w_t1 * (lt1.exp() * (lt1 - ls1)).sum(-1).mean()
    return loss


def kl_vec(p_logits, q_logits, T=2.0):
    lp, lq = F.log_softmax(p_logits.float() / T, -1), F.log_softmax(q_logits.float() / T, -1)
    return (lp.exp() * (lp - lq)).sum(-1)


# ------------------------------------------------------------------------------------------------ targets
def rescale_q(q, y, mu):
    """Decale logit(q_y) d'une constante pour que la CE moyenne vaille mu (garde le classement et la masse hors-y)."""
    ar = np.arange(len(y))
    qy = np.clip(q[ar, y].astype(np.float64), 1e-12, 1 - 1e-12)
    rest = q.astype(np.float64).copy()
    rest[ar, y] = 0
    rest /= np.maximum(rest.sum(1, keepdims=True), 1e-30)
    lg = np.log(qy) - np.log1p(-qy)
    lo, hi = -60.0, 60.0
    for _ in range(100):
        a = (lo + hi) / 2
        if (np.log1p(np.exp(-(lg + a)))).mean() > mu:
            lo = a
        else:
            hi = a
    p = 1 / (1 + np.exp(-(lg + (lo + hi) / 2)))
    out = rest * (1 - p)[:, None]
    out[ar, y] = p
    out = np.maximum(out, 1e-8)
    return (out / out.sum(1, keepdims=True)).astype(np.float32)


def delta_from(z_orig, logq, y):
    """log-odds(vraie classe) de orig moins celui de la cible (>=0) : decalage a appliquer aux vues."""
    ar = torch.arange(len(y), device=y.device)

    def logodds(z):
        z = z.double()
        zm = z.clone()
        zm[ar, y] = -float("inf")
        return z[ar, y] - torch.logsumexp(zm, 1)

    return (logodds(z_orig) - logodds(logq)).clamp_min(0).float()


# ------------------------------------------------------------------------------------------------ trainable selection
def parse_train(spec):
    rules = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        if tok == "all":
            rules.append(lambda n: True)
        elif tok == "head":
            rules.append(lambda n: n.startswith("head."))
        elif tok == "norm":
            rules.append(lambda n: n.startswith("norm.") or n.startswith("fc_norm."))
        else:
            m = re.fullmatch(r"(mlp|fc2|attn|blk|ls|ln)(\d+)-(\d+)", tok)
            if not m:
                raise ValueError(f"--train: token inconnu {tok}")
            kind, a, b = m.group(1), int(m.group(2)), int(m.group(3))

            def rule(n, kind=kind, a=a, b=b):
                parts = n.split(".")
                if parts[0] != "blocks" or not (a <= int(parts[1]) <= b):
                    return False
                sub = parts[2]
                return {"mlp": sub == "mlp", "fc2": n.endswith("mlp.fc2.weight") or n.endswith("mlp.fc2.bias"),
                        "attn": sub == "attn", "blk": True, "ls": sub in ("ls1", "ls2"),
                        "ln": sub in ("norm1", "norm2")}[kind]

            rules.append(rule)
    return lambda n: any(r(n) for r in rules)


# ------------------------------------------------------------------------------------------------ evaluation
def st(t):
    return [round(float(t.mean()), 4), round(float(t.std()), 4)]


@torch.no_grad()
def evaluate(m, C):
    r = {}
    tr, ho = C["tr"], C["ho"]
    for kind, u8 in C["views"].items():
        z = logits_of(m, u8)
        if kind == "medium":  # proxy de non-membres a GARDER : vues ou orig est peu confiant
            lowc = C["lowconf"]
            r["keep_medium"] = {"n": int(lowc.sum()), "kl_orig": round(float(kl_vec(C["z0"][kind][lowc], z[lowc]).mean()), 4),
                                "agree": round(float((z[lowc].argmax(1) == C["z0"][kind][lowc].argmax(1)).float().mean()), 4)}
            if len(C["ho"]):  # vues peu confiantes d'images JAMAIS entrainees : meilleur proxy local du control
                hom = torch.zeros_like(lowc)
                hom[C["ho"]] = True
                mm = lowc & hom
                r["keep_medium"]["ho_n"] = int(mm.sum())
                r["keep_medium"]["ho_kl_orig"] = round(float(kl_vec(C["z0"][kind][mm], z[mm]).mean()), 4)
                r["keep_medium"]["ho_agree"] = round(float((z[mm].argmax(1) == C["z0"][kind][mm].argmax(1)).float().mean()), 4)
            continue
        ce = F.cross_entropy(z, C["y"], reduction="none")
        klo = kl_vec(C["z0"][kind], z)
        gA = C["grpA"]
        e = {"ceA": st(ce[gA]), "ceB": st(ce[~gA]),
             "ce": st(ce[tr]), "acc": round(float((z[tr].argmax(1) == C["y"][tr]).float().mean()), 4),
             "kl_orig": round(float(klo[tr].mean()), 4)}
        if kind == "clean" and C["logq"] is not None:
            e["kl_target"] = round(float(kl_vec(C["logq"][tr], z[tr]).mean()), 4)
            e["target_ce"] = st(F.cross_entropy(C["logq"][tr], C["y"][tr], reduction="none"))
        if len(ho):
            e["ho_kl_orig"] = round(float(klo[ho].mean()), 4)
            e["ho_ce"] = st(ce[ho])
        r[kind] = e
    za = logits_of(m, C["anc_va"])
    kla = float(kl_vec(C["z0_anc"], za).mean())
    agr = float((za.argmax(1) == C["z0_anc"].argmax(1)).float().mean())
    r["anchor"] = {"kl": round(kla, 4), "agree": round(agr, 4)}
    r["retain_est"] = round(agr / (1 + kla), 4)
    km = r["keep_medium"]["kl_orig"]
    r["retain_est2"] = round(1 / (1 + 0.55 * kla + 0.45 * km), 4)
    return r


def fmt_row(tag, cfg, a, r):
    c, mi, he, jp = r["clean"], r["mild"], r["heldout"], r["jpeg"]
    return (f"| {tag} | {cfg} | {a} | {c['ce'][0]:.3f}/{c['ce'][1]:.3f} acc {c['acc']:.3f} | "
            f"{mi['ce'][0]:.3f}/{mi['ce'][1]:.3f} | {he['ce'][0]:.3f}/{he['ce'][1]:.3f} | {jp['ce'][0]:.3f}/{jp['ce'][1]:.3f} | "
            f"{c.get('kl_target', float('nan')):.3f} | {r['anchor']['kl']:.3f}/{r['anchor']['agree']:.3f} | "
            f"{c.get('ho_kl_orig', float('nan')):.3f} | {r['retain_est']:.3f} | "
            f"{r['keep_medium']['kl_orig']:.3f} | {r['retain_est2']:.3f} |\n")


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="")
    ap.add_argument("--target_mu", type=float, default=0.0, help=">0 : recale la CE moyenne de la cible")
    ap.add_argument("--train", default="mlp8-11")
    ap.add_argument("--mask_frac", type=float, default=0.0, help=">0 : ne met a jour que cette fraction (saillance)")
    ap.add_argument("--mask_batches", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--bs_anc", type=int, default=64)
    ap.add_argument("--p_view", type=float, default=0.5)
    ap.add_argument("--views", default="mild", help="familles de vues d'entrainement, ex. mild,heldout")
    ap.add_argument("--view_target", default="shift", choices=["shift", "same"])
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--w_t1", type=float, default=0.5)
    ap.add_argument("--w_anc", type=float, default=3.0)
    ap.add_argument("--w_feat", type=float, default=0.0)
    ap.add_argument("--anc_aug", default="mild")
    ap.add_argument("--anchors", default=str(TASK / "lemasnedechermont1" / "anchors" / "anchors.npz"))
    ap.add_argument("--l2sp", type=float, default=0.0)
    ap.add_argument("--w_keep", type=float, default=0.0, help="KD vers orig sur vues medium du forget peu confiantes")
    ap.add_argument("--keep_kind", default="medium")
    ap.add_argument("--keep_pmax", type=float, default=0.9)
    ap.add_argument("--holdout", type=float, default=0.0)
    ap.add_argument("--alphas", default="0.5,0.75,1.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=int, default=1)
    ap.add_argument("--eval_ckpts", default="")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    dev = "cpu" if a.dry_run else "cuda"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    run = ROOT / "runs" / a.tag
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(vars(a), indent=2))
    lim = 16 if a.dry_run else None

    orig = build()
    orig.load_state_dict(load_sd(TASK / "model.pt"), strict=True)
    orig = orig.to(dev).eval()
    for p in orig.parameters():
        p.requires_grad_(False)
    with np.load(TASK / "forget_images.npz") as d:
        imgs = torch.from_numpy(d["images"][:lim].copy()).to(dev)
    with np.load(a.anchors) as d:
        anc_tr = torch.from_numpy(d["train"][:lim].copy()).to(dev)
        anc_va = torch.from_numpy(d["val"][:lim].copy()).to(dev)
    N = len(imgs)
    views = eval_views(imgs, dev)
    z0 = {k: logits_of(orig, v) for k, v in views.items()}
    z0_anc = logits_of(orig, anc_va)
    y = z0["clean"].argmax(1)
    lowconf = F.softmax(z0["medium"], -1).gather(1, y[:, None]).squeeze(1) < a.keep_pmax
    print(f"vues medium : {int(lowconf.sum())}/{N} avec p_orig(vraie) < {a.keep_pmax} (proxy non-membres)", flush=True)
    logq = None
    if a.targets:
        with np.load(a.targets) as d:
            q, yq = d["q"][:lim].astype(np.float32), d["y"][:lim].astype(np.int64)
        agree_y = float((torch.from_numpy(yq).to(dev) == y).float().mean())
        print(f"labels cible == argmax orig : {agree_y:.4f}", flush=True)
        y = torch.from_numpy(yq).to(dev)
        if a.target_mu > 0:
            q = rescale_q(q, yq, a.target_mu)
        logq = torch.log(torch.from_numpy(q).to(dev).clamp_min(1e-12))
    if a.holdout > 0:
        perm = np.random.default_rng(1234).permutation(N)
        k = int(round(N * (1 - a.holdout)))
        tr_idx, ho_idx = np.sort(perm[:k]), np.sort(perm[k:])
    else:
        tr_idx, ho_idx = np.arange(N), np.array([], dtype=np.int64)
    cnt = torch.bincount(y, minlength=100)
    grpA = cnt[y] >= 129
    print(f"groupe A (classes pleines) : {int(grpA.sum())} images", flush=True)
    C = {"grpA": grpA, "lowconf": lowconf, "views": views, "z0": z0, "z0_anc": z0_anc, "anc_va": anc_va, "y": y, "logq": logq,
         "tr": torch.as_tensor(tr_idx, device=dev), "ho": torch.as_tensor(ho_idx, device=dev)}
    print(f"setup {time.time() - t0:.0f}s  N={N} anchors {len(anc_tr)}/{len(anc_va)}", flush=True)

    res_md = ROOT / "RESULTS_LOCAL4.md"
    if not res_md.exists():
        res_md.write_text("| tag | config | alpha | clean CE m/s | mild CE | heldout CE | jpeg CE (jamais vu) | KL cible | ancres KL/accord | holdout KL | retain_est | KL vues medium peu confiantes | retain_est2 |\n"
                          "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")

    # -------------------------------------------------------------- mode mesure seule
    if a.eval_ckpts:
        evalm = copy.deepcopy(orig)
        out = {}
        for path in [p for p in a.eval_ckpts.split(",") if p]:
            evalm.load_state_dict(load_sd(path), strict=True)
            r = evaluate(evalm, C)
            out[path] = r
            print("EVAL", Path(path).name, json.dumps(r), flush=True)
            with open(res_md, "a") as f:
                f.write(fmt_row(a.tag, "eval " + Path(path).name, "-", r))
        (run / "eval.json").write_text(json.dumps(out, indent=2))
        return

    assert logq is not None, "--targets requis pour entrainer"
    delta = delta_from(z0["clean"], logq, y)

    # -------------------------------------------------------------- student
    student = copy.deepcopy(orig)
    sel = parse_train(a.train)
    params = []
    for n, p in student.named_parameters():
        p.requires_grad_(bool(sel(n)))
        if p.requires_grad:
            params.append((n, p))
    ntr = sum(p.numel() for _, p in params)
    print(f"trainable {ntr / 1e6:.2f}M in {len(params)} tensors ({a.train})", flush=True)
    student.train()

    if a.mask_frac > 0:  # masque de saillance facon SalUn sur la perte forget
        acc = {n: torch.zeros_like(p) for n, p in params}
        for b, i in enumerate(range(0, len(tr_idx), a.bs)):
            if b >= a.mask_batches:
                break
            ib = torch.as_tensor(tr_idx[i:i + a.bs], device=dev)
            loss = kd(logq[ib], fwd(student, u8_to01(imgs[ib])), a.T, a.w_t1)
            student.zero_grad(set_to_none=True)
            loss.backward()
            for n, p in params:
                acc[n] += p.grad.abs()
        kept = 0
        for n, p in params:
            g = acc[n].flatten()
            k = int(round(g.numel() * (1 - a.mask_frac)))
            thr = g.kthvalue(max(1, min(k, g.numel()))).values
            m = (acc[n] >= thr).float()
            kept += int(m.sum())
            p.register_hook(lambda gr, m=m: gr * m)
        student.zero_grad(set_to_none=True)
        print(f"mask: {kept / ntr:.3f} of trainable weights kept", flush=True)

    ref = {n: p.detach().clone() for n, p in params} if a.l2sp > 0 else None
    opt = torch.optim.AdamW([p for _, p in params], lr=a.lr, weight_decay=0.0)
    bs = 4 if a.dry_run else a.bs
    bs_anc = 4 if a.dry_run else a.bs_anc
    epochs = 1 if a.dry_run else a.epochs
    total = math.ceil(len(tr_idx) / bs) * epochs
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    gen = make_gen(dev, a.seed + 1)
    rng = np.random.default_rng(a.seed)
    vkinds = [v for v in a.views.split(",") if v]
    onehot = F.one_hot(y, 100).float()
    step = 0
    for ep in range(epochs):
        perm = rng.permutation(tr_idx)
        for i in range(0, len(perm), bs):
            ib = torch.as_tensor(perm[i:i + bs], device=dev)
            x = u8_to01(imgs[ib])
            use_v = torch.rand(len(ib), device=dev, generator=gen) < a.p_view
            if vkinds and bool(use_v.any()):
                xv = augment(x, vkinds[int(rng.integers(len(vkinds)))], gen)
                xin = torch.where(use_v.view(-1, 1, 1, 1), xv, x)
            else:
                xin, use_v = x, torch.zeros_like(use_v)
            with torch.no_grad():
                teacher = logq[ib]
                if a.view_target == "shift" and bool(use_v.any()):
                    tv = fwd(orig, xin) - delta[ib, None] * onehot[ib]
                    teacher = torch.where(use_v.view(-1, 1), tv, teacher)
            loss_f = kd(teacher, fwd(student, xin), a.T, a.w_t1)
            loss = loss_f
            la = lfe = torch.zeros((), device=dev)
            if a.w_anc > 0 or a.w_feat > 0:
                ai = torch.randint(0, len(anc_tr), (bs_anc,), device=dev, generator=gen)
                xa = augment(u8_to01(anc_tr[ai]), a.anc_aug, gen)
                with torch.no_grad():
                    ta, tpre = fwd(orig, xa, feat=True)
                sa, spre = fwd(student, xa, feat=True)
                if a.w_anc > 0:
                    la = kd(ta, sa, a.T)
                    loss = loss + a.w_anc * la
                if a.w_feat > 0:
                    lfe = F.mse_loss(spre, tpre)
                    loss = loss + a.w_feat * lfe
            lk = torch.zeros((), device=dev)
            if a.w_keep > 0:
                with torch.no_grad():
                    xk = augment(x, a.keep_kind, gen)
                    tk = fwd(orig, xk)
                    selk = F.softmax(tk, -1).gather(1, y[ib][:, None]).squeeze(1) < a.keep_pmax
                if bool(selk.any()):
                    lk = kd(tk[selk], fwd(student, xk[selk]), a.T)
                    loss = loss + a.w_keep * lk
            if ref is not None:
                loss = loss + a.l2sp * sum(((p - ref[n]) ** 2).sum() for n, p in params)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in params], 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 25 == 0 or a.dry_run:
                print(f"ep {ep} step {step}/{total} forget {loss_f.item():.4f} anc {la.item():.4f} "
                      f"feat {lfe.item():.4f} keep {lk.item():.4f} {time.time() - t0:.0f}s", flush=True)
            if a.dry_run and step >= 2:
                break
    student.eval()

    # -------------------------------------------------------------- interpolation + mesures
    theta0 = {k: v.detach().clone() for k, v in orig.state_dict().items()}
    theta1 = {k: v.detach().clone() for k, v in student.state_dict().items()}
    evalm = copy.deepcopy(orig)
    cfg = (f"{a.train} m{a.mask_frac:g} lr{a.lr:g} ep{a.epochs} wa{a.w_anc:g} wf{a.w_feat:g} wk{a.w_keep:g} l2{a.l2sp:g} "
           f"v{a.views}/{a.view_target} mu{a.target_mu:g} {Path(a.targets).stem}")
    results = {}
    for al in [float(s) for s in a.alphas.split(",")]:
        sd = {k: (theta0[k] + al * (theta1[k] - theta0[k])) if theta0[k].is_floating_point() else theta1[k]
              for k in theta0}
        evalm.load_state_dict(sd, strict=True)
        r = evaluate(evalm, C)
        path = run / f"alpha_{al:g}.pt"
        if a.save and not a.dry_run:
            torch.save({k: v.detach().cpu().float().contiguous() for k, v in sd.items()}, path)
            r["path"] = str(path)
        results[f"{al:g}"] = r
        print("RESULT", f"alpha={al:g}", json.dumps(r), flush=True)
        if not a.dry_run:
            with open(res_md, "a") as f:
                f.write(fmt_row(a.tag, cfg, f"{al:g}", r))
    (run / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
