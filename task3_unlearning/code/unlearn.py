"""Targeted soft-label distillation unlearning. One run -> one .pt per interpolation alpha + metrics."""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (ANCHORS, DEVICE, OUT, HERE, augment, dump_json, load_images_gpu, load_orig, make_gen, norm,
                    save_state_dict, seed_all, u8_to01)
from evaluate_local import Ctx, evaluate, mix_batch

ap = argparse.ArgumentParser()
ap.add_argument("--targets", required=True)
ap.add_argument("--folds", default="all", help="all | 80 | k5_<k> (k-th of 5 folds held out)")
ap.add_argument("--w_ho", type=float, default=0.0, help="KD to orig on the held-out fold (instance-specific forgetting)")
ap.add_argument("--trainable", default="all", choices=["all", "last4", "lsnh", "head"])
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--p_view", type=float, default=0.5, help="fraction of samples replaced by a mild view")
ap.add_argument("--T", type=float, default=2.0)
ap.add_argument("--w_t1", type=float, default=0.5, help="extra KL weight at T=1 (CE scale)")
ap.add_argument("--l2sp", type=float, default=0.0)
ap.add_argument("--w_mix", type=float, default=0.0, help="retain-proxy: distill to orig on mixes")
ap.add_argument("--w_anchor", type=float, default=0.0, help="retain-proxy: distill to orig on public anchor photos")
ap.add_argument("--alphas", default="0.5,0.75,1.0,1.25")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--save", type=int, default=1)
ap.add_argument("--tag", required=True)
ap.add_argument("--init", default="", help="start the student from this checkpoint instead of orig")
args = ap.parse_args()

seed_all(args.seed)
run_dir = OUT / args.tag
run_dir.mkdir(parents=True, exist_ok=True)
dump_json(vars(args), run_dir / "config.json")

orig = load_orig()
for p in orig.parameters():
    p.requires_grad_(False)
imgs = load_images_gpu()
ctx = Ctx(orig, imgs, args.targets, args.folds)
theta0 = {k: v.detach().clone() for k, v in orig.state_dict().items()}
anc_tr = torch.as_tensor(np.load(ANCHORS)["train"], device=DEVICE) if args.w_anchor > 0 else None

student = copy.deepcopy(orig).train()
if args.init:  # e.g. repair an extrapolated model; alpha-interpolation below stays relative to orig
    student.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=True), strict=True)
    student = student.to(DEVICE).train()


def is_trainable(name):
    if args.trainable == "all":
        return True
    if args.trainable == "head":
        return name.startswith("head.") or name.startswith("fc_norm.")
    if args.trainable == "lsnh":
        return (".ls1." in name or ".ls2." in name or "norm" in name or name.startswith("head."))
    if args.trainable == "last4":
        if name.startswith("blocks."):
            return int(name.split(".")[1]) >= 8
        return name.startswith("norm.") or name.startswith("head.") or name.startswith("fc_norm.")


params = []
for n_, p in student.named_parameters():
    p.requires_grad_(is_trainable(n_))
    if p.requires_grad:
        params.append((n_, p))
print(f"trainable params: {sum(p.numel() for _, p in params) / 1e6:.2f}M in {len(params)} tensors")
ref = {n_: p.detach().clone() for n_, p in params}
opt = torch.optim.AdamW([p for _, p in params], lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))

tr = ctx.tr
steps_per_epoch = math.ceil(len(tr) / args.bs)
total = steps_per_epoch * args.epochs
warm = max(1, int(0.1 * total))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
gen = make_gen(args.seed + 1)
rng = np.random.default_rng(args.seed)
T = args.T


def kd(teacher, student_logits):
    lt = F.log_softmax(teacher / T, -1)
    ls = F.log_softmax(student_logits.float() / T, -1)
    loss = (lt.exp() * (lt - ls)).sum(-1).mean() * T * T
    if args.w_t1 > 0:
        lt1 = F.log_softmax(teacher, -1)
        ls1 = F.log_softmax(student_logits.float(), -1)
        loss = loss + args.w_t1 * (lt1.exp() * (lt1 - ls1)).sum(-1).mean()
    return loss


t0 = time.time()
step = 0
log = []
for ep in range(args.epochs):
    perm = rng.permutation(tr)
    for i in range(0, len(perm), args.bs):
        idx = torch.as_tensor(perm[i:i + args.bs], device=DEVICE)
        x = u8_to01(imgs[idx])
        y = ctx.y[idx]
        use_view = torch.rand(len(idx), device=DEVICE, generator=gen) < args.p_view
        xv = augment(x, "mild", gen)
        xin = torch.where(use_view.view(-1, 1, 1, 1), xv, x)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            t_view = orig(norm(xin)).float()
        t_view = t_view - ctx.delta[idx, None] * F.one_hot(y, 100).float()
        teacher = torch.where(use_view.view(-1, 1), t_view, ctx.logq[idx])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = student(norm(xin))
        loss = kd(teacher, s)
        lm = torch.zeros((), device=DEVICE)
        if args.w_mix > 0:
            xm = mix_batch(x, x.roll(1, 0), gen)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                tm = orig(norm(xm)).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sm = student(norm(xm))
            lm = kd(tm, sm)
            loss = loss + args.w_mix * lm
        if args.w_ho > 0 and len(ctx.ho):
            hi = torch.as_tensor(rng.choice(ctx.ho, args.bs // 2), device=DEVICE)
            xh = u8_to01(imgs[hi])
            xh = torch.where((torch.rand(len(hi), device=DEVICE, generator=gen) < args.p_view).view(-1, 1, 1, 1),
                             augment(xh, "mild", gen), xh)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                th = orig(norm(xh)).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sh = student(norm(xh))
            loss = loss + args.w_ho * kd(th, sh)
        if anc_tr is not None:
            ai = torch.randint(0, len(anc_tr), (args.bs,), device=DEVICE, generator=gen)
            xa = augment(u8_to01(anc_tr[ai]), "mild", gen)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ta = orig(norm(xa)).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sa = student(norm(xa))
            loss = loss + args.w_anchor * kd(ta, sa)
        if args.l2sp > 0:
            loss = loss + args.l2sp * sum(((p - ref[n_]) ** 2).sum() for n_, p in params)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for _, p in params], 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 20 == 0:
            print(f"ep {ep} step {step}/{total} loss {loss.item():.4f} mix {lm.item():.4f} {time.time() - t0:.0f}s",
                  flush=True)
student.eval()
theta1 = {k: v.detach().clone() for k, v in student.state_dict().items()}

results = {}
evalm = copy.deepcopy(orig)
for a in [float(s) for s in args.alphas.split(",")]:
    sd = {k: (theta0[k] + a * (theta1[k] - theta0[k])) if theta0[k].is_floating_point() else theta1[k] for k in theta0}
    evalm.load_state_dict(sd, strict=True)
    r = evaluate(evalm, ctx)
    path = run_dir / f"alpha_{a:g}.pt"
    if args.save:
        save_state_dict(sd, path)
    r["path"] = str(path)
    results[f"{a:g}"] = r
    c, h = r["clean"], r["heldout_alt"]
    rp = r.get("foldout", r["mix"])
    print(f"alpha {a:g}: clean ce {c['ce_mean']:.3f}/{c['ce_std']:.3f} (gold {c['gold_ce_mean']:.3f}/{c['gold_ce_std']:.3f})"
          f" acc {c['acc']:.3f} | held ce {h['ce_mean']:.3f}/{h['ce_std']:.3f} (gold {h['gold_ce_mean']:.3f}/"
          f"{h['gold_ce_std']:.3f}) | forget_proxy {r['forget_proxy']:.3f} retain agree {rp['agree']:.4f} "
          f"kl {rp['kl']:.4f} | final_proxy {r['final_proxy']:.3f}"
          + f" | T2 entropy clean {c['t2_entropy']:.2f} held {h['t2_entropy']:.2f} ptrue {c['t2_ptrue']:.3f}"
          + (f" | anchor agree {r['anchor']['agree']:.4f} kl {r['anchor']['kl']:.4f}" if "anchor" in r else "")
          + (f" | foldout agree {r['foldout']['agree']:.4f} kl {r['foldout']['kl']:.4f}" if "foldout" in r else ""),
          flush=True)
    for g in ("clean_kept", "clean_removed"):
        if g in r:
            print(f"    {g}: n={r[g]['n']} ce {r[g]['ce_mean']:.3f}/{r[g]['ce_std']:.3f} (gold {r[g]['gold_ce_mean']:.3f})"
                  f" acc {r[g]['acc']:.3f}", flush=True)
dump_json(results, run_dir / "metrics.json")

with open(HERE / "RESULTS.md", "a") as f:
    for a, r in results.items():
        c, h = r["clean"], r["heldout_alt"]
        rp = r.get("foldout", r["mix"])
        f.write(f"| {args.tag} | {Path(args.targets).stem} | {args.folds} | {args.trainable} | {args.lr:g} | "
                f"{args.epochs} | {args.l2sp:g} | {args.w_mix:g} | {a} | {c['ce_mean']:.3f}/{c['ce_std']:.3f} | "
                f"{h['ce_mean']:.3f}/{h['ce_std']:.3f} | {r['forget_proxy']:.3f} | {rp['agree']:.4f}/{rp['kl']:.4f} | "
                f"{r['final_proxy']:.3f} | {r['path'] if args.save else '-'} | |\n")
print("done", time.time() - t0)
