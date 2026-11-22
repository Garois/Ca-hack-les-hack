"""Local proxy scoring. 'gold' is replaced by our targets (clean: q; heldout alteration family: orig(view) - delta*e_y).
Retain proxies: (a) fold-out 20% of forget images not unlearned (only when trained with --folds 80),
(b) cutmix/mixup composites of forget images. Both vs orig: KL at T=2 and argmax agreement."""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from common import (ART, DEVICE, ce_vec, fold_split, kl_vec, load_images_gpu, load_orig, logits_of, make_gen, norm,
                    u8_to01, tt)

HELD_SEED = 4242
MIX_SEED = 777


@torch.no_grad()
def mix_logits(model, imgs, idx, seed=MIX_SEED, bs=128):
    model.eval()
    gen = make_gen(seed)
    idx = np.asarray(idx)
    out = []
    for i in range(0, len(idx), bs):
        a = u8_to01(imgs[torch.as_tensor(idx[i:i + bs], device=DEVICE)])
        b = a.roll(1, 0)
        x = mix_batch(a, b, gen)
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(norm(x)).float())
    return torch.cat(out)


def mix_batch(a, b, gen):
    B = a.shape[0]
    lam = torch.rand(B, device=a.device, generator=gen) * 0.4 + 0.3
    use_cut = torch.rand(B, device=a.device, generator=gen) < 0.5
    mixed = lam.view(B, 1, 1, 1) * a + (1 - lam.view(B, 1, 1, 1)) * b
    H = a.shape[-1]
    cut = a.clone()
    side = int(H * 0.55)
    off = int(torch.randint(0, H - side, (1,), device=a.device, generator=gen))
    cut[:, :, off:off + side, off:off + side] = b[:, :, off:off + side, off:off + side]
    x = torch.where(use_cut.view(B, 1, 1, 1), cut, mixed)
    return torch.round(x * 255) / 255


class Ctx:
    """Precomputes everything that depends only on orig + targets."""

    def __init__(self, orig, imgs, tgt_path, folds="all"):
        T = np.load(tgt_path)
        self.imgs = imgs
        self.y = torch.as_tensor(T["y"], device=DEVICE).long()
        self.logq = torch.log(torch.as_tensor(T["q"], device=DEVICE).float())
        self.delta = torch.as_tensor(T["delta"], device=DEVICE).float()
        rem = T["removed"] if "removed" in T.files else np.zeros(len(T["y"]), bool)
        self.removed = torch.as_tensor(rem, device=DEVICE)
        n = len(imgs)
        if folds == "80":
            self.tr, self.ho = fold_split(n)
        elif folds.startswith("k5_"):
            from common import kfold_split
            self.tr, self.ho = kfold_split(n, int(folds[3:]))
        else:
            self.tr, self.ho = np.arange(n), np.array([], dtype=int)
        self.ev = self.tr  # forget metrics on the images being unlearned
        self.orig_clean = logits_of(orig, imgs, self.ev)
        self.orig_held = logits_of(orig, imgs, self.ev, kind="heldout", seed=HELD_SEED)
        ev_t = torch.as_tensor(self.ev, device=DEVICE)
        self.gold_held = self.orig_held - self.delta[ev_t, None] * F.one_hot(self.y[ev_t], 100).float()
        self.orig_ho = logits_of(orig, imgs, self.ho) if len(self.ho) else None
        mix_idx = self.ho if len(self.ho) else self.ev[:1000]
        self.mix_idx = mix_idx
        self.orig_mix = mix_logits(orig, imgs, mix_idx)
        from common import ANCHORS
        self.anc = torch.as_tensor(np.load(ANCHORS)["val"], device=DEVICE) if ANCHORS.exists() else None
        self.orig_anc = logits_of(orig, self.anc) if self.anc is not None else None


def _group(sub, orig, gold_logits, y):
    Ls, Lo, Lg = ce_vec(sub, y), ce_vec(orig, y), ce_vec(gold_logits, y)
    d_sub = (Ls.mean() - Lg.mean()).abs() + (Ls.std() - Lg.std()).abs()
    d_orig = (Lo.mean() - Lg.mean()).abs() + (Lo.std() - Lg.std()).abs()
    ms = 1 - float((d_sub / d_orig.clamp_min(1e-8)).clamp(0, 1))
    kl_s = kl_vec(gold_logits, sub).mean()
    kl_o = kl_vec(gold_logits, orig).mean()
    pr = 1 - float((kl_s / kl_o.clamp_min(1e-8)).clamp(0, 1))
    p2 = F.softmax(sub.float() / 2, -1)
    return {"t2_entropy": float(-(p2 * torch.log(p2 + 1e-30)).sum(-1).mean()),
            "t2_ptrue": float(p2.gather(1, y[:, None]).mean()),
            "ce_mean": float(Ls.mean()), "ce_std": float(Ls.std()), "gold_ce_mean": float(Lg.mean()),
            "gold_ce_std": float(Lg.std()), "acc": float((sub.argmax(1) == y).float().mean()),
            "gold_acc": float((gold_logits.argmax(1) == y).float().mean()),
            "argmax_eq_gold": float((sub.argmax(1) == gold_logits.argmax(1)).float().mean()),
            "kl_gold_sub": float(kl_s), "mean_std_score": ms, "pred_score": pr}


@torch.no_grad()
def evaluate(model, ctx):
    ev_t = torch.as_tensor(ctx.ev, device=DEVICE)
    y = ctx.y[ev_t]
    sub_clean = logits_of(model, ctx.imgs, ctx.ev)
    sub_held = logits_of(model, ctx.imgs, ctx.ev, kind="heldout", seed=HELD_SEED)
    g1 = _group(sub_clean, ctx.orig_clean, ctx.logq[ev_t], y)
    g2 = _group(sub_held, ctx.orig_held, ctx.gold_held, y)
    forget = ((g1["mean_std_score"] + g2["mean_std_score"]) / 2) * ((g1["pred_score"] + g2["pred_score"]) / 2)
    res = {"clean": g1, "heldout_alt": g2, "forget_proxy": forget}
    rm = ctx.removed[ev_t]
    ce_s, ce_g = ce_vec(sub_clean, y), ce_vec(ctx.logq[ev_t], y)
    for name, m in (("removed", rm), ("kept", ~rm)):
        if m.any():
            res[f"clean_{name}"] = {"n": int(m.sum()), "ce_mean": float(ce_s[m].mean()), "ce_std": float(ce_s[m].std()),
                                    "gold_ce_mean": float(ce_g[m].mean()),
                                    "acc": float((sub_clean[m].argmax(1) == y[m]).float().mean())}
    sub_mix = mix_logits(model, ctx.imgs, ctx.mix_idx)
    kl_mix = float(kl_vec(ctx.orig_mix, sub_mix).mean())
    res["mix"] = {"kl": kl_mix, "agree": float((sub_mix.argmax(1) == ctx.orig_mix.argmax(1)).float().mean())}
    if ctx.anc is not None:
        sub_anc = logits_of(model, ctx.anc)
        res["anchor"] = {"kl": float(kl_vec(ctx.orig_anc, sub_anc).mean()),
                         "agree": float((sub_anc.argmax(1) == ctx.orig_anc.argmax(1)).float().mean())}
    if ctx.orig_ho is not None:
        sub_ho = logits_of(model, ctx.imgs, ctx.ho)
        res["foldout"] = {"kl": float(kl_vec(ctx.orig_ho, sub_ho).mean()),
                          "agree": float((sub_ho.argmax(1) == ctx.orig_ho.argmax(1)).float().mean())}
        r = res["foldout"]
    else:
        r = res["mix"]
    res["retain_proxy"] = r["agree"] / (1 + r["kl"])
    res["final_proxy"] = forget * res["retain_proxy"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--folds", default="all", choices=["all", "80"])
    a = ap.parse_args()
    orig = load_orig()
    imgs = load_images_gpu()
    ctx = Ctx(orig, imgs, a.targets, a.folds)
    m = tt.build_model()
    m.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True), strict=True)
    m = m.to(DEVICE).eval()
    print(json.dumps(evaluate(m, ctx), indent=2))


if __name__ == "__main__":
    main()
