"""v2: soft-label distillation on kept forget images + anchor KD to orig on public photos (+ removed images),
restricted trainable params, L2-SP. Then optional head-bias surgery (--D on classes with count >= --cap).
Writes one .pt per alpha (interpolation orig -> trained, surgery applied after) + metrics."""
import argparse
import copy
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import (ANCHORS, ART, DEVICE, HERE, OUT, augment, ce_vec, dump_json, fold_split, kl_vec, load_images_gpu,
                    load_orig, logits_of, make_gen, norm, save_state_dict, seed_all, u8_to01)

ap = argparse.ArgumentParser()
ap.add_argument("--targets", required=True)
ap.add_argument("--trainable", default="last4", choices=["all", "last2", "last4", "last6", "head"])
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--p_view", type=float, default=0.5)
ap.add_argument("--T", type=float, default=2.0)
ap.add_argument("--w_t1", type=float, default=0.5)
ap.add_argument("--w_anchor", type=float, default=1.0)
ap.add_argument("--w_rm", type=float, default=0.5, help="KD to orig on removed-class images")
ap.add_argument("--l2sp", type=float, default=0.0)
ap.add_argument("--holdout", type=float, default=0.0, help="fraction of kept images held out (diagnostic)")
ap.add_argument("--D", type=float, default=0.0, help="head-bias surgery strength for removed classes")
ap.add_argument("--rm_ce", type=float, default=0.0, help="if >0: solve D per alpha so median removed CE == rm_ce")
ap.add_argument("--cap", type=int, default=129)
ap.add_argument("--alphas", default="0.5,0.75,1.0")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--save", type=int, default=1)
ap.add_argument("--tag", required=True)
args = ap.parse_args()

seed_all(args.seed)
run_dir = OUT / args.tag
run_dir.mkdir(parents=True, exist_ok=True)
dump_json(vars(args), run_dir / "config.json")

orig = load_orig()
for p in orig.parameters():
    p.requires_grad_(False)
imgs = load_images_gpu()
Tz = np.load(args.targets)
y = torch.as_tensor(Tz["y"], device=DEVICE).long()
logq = torch.log(torch.as_tensor(Tz["q"], device=DEVICE).float())
removed = Tz["removed"]
S = np.load(ART / "orig_stats.npz")
counts = S["counts"]
rm_cls = torch.as_tensor(np.where(counts >= args.cap)[0], device=DEVICE)

kept_idx = np.where(~removed)[0]
rm_idx = np.where(removed)[0]
if args.holdout > 0:
    a, b = fold_split(len(kept_idx), 1 - args.holdout, seed=99)
    tr_idx, ho_idx = kept_idx[a], kept_idx[b]
else:
    tr_idx, ho_idx = kept_idx, np.array([], dtype=int)
A = np.load(ANCHORS)
anc_tr = torch.as_tensor(A["train"], device=DEVICE)
anc_va = torch.as_tensor(A["val"], device=DEVICE)

student = copy.deepcopy(orig).train()
K = {"last2": 10, "last4": 8, "last6": 6}.get(args.trainable)


def is_trainable(name):
    if args.trainable == "all":
        return True
    if name.startswith("head.") or name.startswith("norm.") or name.startswith("fc_norm."):
        return True
    if args.trainable == "head":
        return False
    return name.startswith("blocks.") and int(name.split(".")[1]) >= K


params = []
for n_, p in student.named_parameters():
    p.requires_grad_(is_trainable(n_))
    if p.requires_grad:
        params.append((n_, p))
print(f"trainable {sum(p.numel() for _, p in params) / 1e6:.1f}M; kept train {len(tr_idx)} holdout {len(ho_idx)} "
      f"removed {len(rm_idx)} anchors {len(anc_tr)}", flush=True)
ref = {n_: p.detach().clone() for n_, p in params}
opt = torch.optim.AdamW([p for _, p in params], lr=args.lr, weight_decay=0.0)
steps_per_epoch = math.ceil(len(tr_idx) / args.bs)
total = steps_per_epoch * args.epochs
warm = max(1, int(0.1 * total))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
gen = make_gen(args.seed + 1)
rng = np.random.default_rng(args.seed)
T = args.T


def kd(t_logits, s_logits, w_t1):
    lt, ls = F.log_softmax(t_logits / T, -1), F.log_softmax(s_logits.float() / T, -1)
    loss = (lt.exp() * (lt - ls)).sum(-1).mean() * T * T
    if w_t1 > 0:
        lt1, ls1 = F.log_softmax(t_logits, -1), F.log_softmax(s_logits.float(), -1)
        loss = loss + w_t1 * (lt1.exp() * (lt1 - ls1)).sum(-1).mean()
    return loss


def fwd(m, x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return m(norm(x)).float()


t0 = time.time()
step = 0
for ep in range(args.epochs):
    perm = rng.permutation(tr_idx)
    for i in range(0, len(perm), args.bs):
        idx = torch.as_tensor(perm[i:i + args.bs], device=DEVICE)
        x = u8_to01(imgs[idx])
        use_view = (torch.rand(len(idx), device=DEVICE, generator=gen) < args.p_view).view(-1, 1, 1, 1)
        x = torch.where(use_view, augment(x, "mild", gen), x)
        loss_f = kd(logq[idx], fwd(student, x), args.w_t1)
        loss = loss_f
        la = torch.zeros((), device=DEVICE)
        if args.w_anchor > 0:
            ai = torch.randint(0, len(anc_tr), (args.bs,), device=DEVICE, generator=gen)
            xa = augment(u8_to01(anc_tr[ai]), "mild", gen)
            with torch.no_grad():
                ta = fwd(orig, xa)
            la = kd(ta, fwd(student, xa), 0.0)
            loss = loss + args.w_anchor * la
        lr_ = torch.zeros((), device=DEVICE)
        if args.w_rm > 0 and len(rm_idx):
            ri = torch.as_tensor(rng.choice(rm_idx, args.bs // 2), device=DEVICE)
            xr = u8_to01(imgs[ri])
            xr = torch.where((torch.rand(len(ri), device=DEVICE, generator=gen) < 0.5).view(-1, 1, 1, 1),
                             augment(xr, "mild", gen), xr)
            with torch.no_grad():
                tr_ = fwd(orig, xr)
            lr_ = kd(tr_, fwd(student, xr), 0.0)
            loss = loss + args.w_rm * lr_
        if args.l2sp > 0:
            loss = loss + args.l2sp * sum(((p - ref[n_]) ** 2).sum() for n_, p in params)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for _, p in params], 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 25 == 0:
            print(f"ep {ep} step {step}/{total} forget {loss_f.item():.4f} anchor {la.item():.4f} "
                  f"rm {lr_.item():.4f} {time.time() - t0:.0f}s", flush=True)
student.eval()

# ------------------------------------------------------------------ evaluation per alpha
theta0 = {k: v.detach().clone() for k, v in orig.state_dict().items()}
theta1 = {k: v.detach().clone() for k, v in student.state_dict().items()}
o_clean = logits_of(orig, imgs)
o_held = logits_of(orig, imgs, kind="heldout", seed=4242)
o_anc = logits_of(orig, anc_va)
o_ho = o_clean[torch.as_tensor(ho_idx, device=DEVICE)] if len(ho_idx) else None
qT = logq
kt = torch.as_tensor(tr_idx, device=DEVICE)
rmt = torch.as_tensor(rm_idx, device=DEVICE)


def st(t):
    return [round(float(t.mean()), 3), round(float(t.std()), 3)]


evalm = copy.deepcopy(orig)
results = {}
for a in [float(s) for s in args.alphas.split(",")]:
    sd = {k: (theta0[k] + a * (theta1[k] - theta0[k])) if theta0[k].is_floating_point() else theta1[k] for k in theta0}
    evalm.load_state_dict(sd, strict=True)
    s_clean = logits_of(evalm, imgs)
    s_held = logits_of(evalm, imgs, kind="heldout", seed=4242)
    s_anc = logits_of(evalm, anc_va)
    anc_pre = {"agree": round(float((s_anc.argmax(1) == o_anc.argmax(1)).float().mean()), 4),
               "kl": round(float(kl_vec(o_anc, s_anc).mean()), 4)}
    D = args.D
    shift = torch.zeros(100, device=DEVICE)
    shift[rm_cls] = 1.0
    if args.rm_ce > 0 and len(rm_idx):
        lo, hi = 0.0, 60.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if ce_vec(s_clean[rmt] - mid * shift, y[rmt]).median() < args.rm_ce:
                lo = mid
            else:
                hi = mid
        D = (lo + hi) / 2
    if D > 0:  # bias shift == exact logit shift
        sd["head.bias"] = sd["head.bias"].clone()
        sd["head.bias"][rm_cls] -= D
        s_clean, s_held, s_anc = s_clean - D * shift, s_held - D * shift, s_anc - D * shift
    ce_c, ce_h = ce_vec(s_clean, y), ce_vec(s_held, y)
    r = {"alpha": a, "D": round(D, 3), "anchor_pre_surgery": anc_pre,
         "kept_clean_ce": st(ce_c[kt]), "kept_target_ce": st(ce_vec(qT[kt], y[kt])),
         "kept_held_ce": st(ce_h[kt]), "kept_acc": round(float((s_clean[kt].argmax(1) == y[kt]).float().mean()), 3),
         "kept_kl_q_sub": round(float(kl_vec(qT[kt], s_clean[kt]).mean()), 4),
         "kept_kl_q_orig": round(float(kl_vec(qT[kt], o_clean[kt]).mean()), 4),
         "all_clean_ce": st(ce_c), "all_held_ce": st(ce_h),
         "anchor_agree": round(float((s_anc.argmax(1) == o_anc.argmax(1)).float().mean()), 4),
         "anchor_kl": round(float(kl_vec(o_anc, s_anc).mean()), 4)}
    if len(rm_idx):
        r["rm_clean_ce"] = st(ce_c[rmt])
        r["rm_held_ce"] = st(ce_h[rmt])
    if o_ho is not None:
        hot = torch.as_tensor(ho_idx, device=DEVICE)
        r["holdout_kl_orig"] = round(float(kl_vec(o_ho, s_clean[hot]).mean()), 4)
        r["holdout_agree"] = round(float((s_clean[hot].argmax(1) == o_ho.argmax(1)).float().mean()), 4)
        r["holdout_ce"] = st(ce_c[hot])
    path = run_dir / f"alpha_{a:g}.pt"
    if args.save:
        save_state_dict(sd, path)
    r["path"] = str(path)
    results[f"{a:g}"] = r
    print("RESULT " + json.dumps(r), flush=True)
dump_json(results, run_dir / "metrics.json")
with open(HERE / "RESULTS.md", "a") as f:
    for a, r in results.items():
        f.write(f"| {args.tag} | {args.targets.split('/')[-1]} | {args.trainable} lr{args.lr:g} ep{args.epochs} "
                f"wa{args.w_anchor:g} l2sp{args.l2sp:g} D{args.D:g} | {a} | kept {r['kept_clean_ce']} tgt "
                f"{r['kept_target_ce']} | all {r['all_clean_ce']} held {r['all_held_ce']} | anchor "
                f"{r['anchor_agree']}/{r['anchor_kl']} | {r['path'] if args.save else '-'} | |\n")
print("done", round(time.time() - t0))
