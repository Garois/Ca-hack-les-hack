"""P0: analysis of orig on the forget set + difficulty proxies. Writes artifacts/orig_stats.npz and artifacts/analysis.json"""
import numpy as np
import torch
import torch.nn.functional as F

from common import ART, DEVICE, ce_vec, dump_json, load_images_gpu, load_orig, logits_of, seed_all

N_STRONG = 16
N_MILD = 8


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def mid_features(model, imgs, blocks=(6, 7, 8, 9), bs=128):
    feats, store = [], {}
    hooks = [model.blocks[b].register_forward_hook(lambda m, i, o, b=b: store.__setitem__(b, o)) for b in blocks]
    from common import u8_to01, norm
    for i in range(0, len(imgs), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            model(norm(u8_to01(imgs[i:i + bs])))
        cls = torch.stack([store[b][:, 0].float() for b in blocks]).mean(0)
        pat = torch.stack([store[b][:, 1:].float().mean(1) for b in blocks]).mean(0)
        feats.append(torch.cat([cls, pat], 1))
    for h in hooks:
        h.remove()
    return torch.cat(feats)


def probe_oof(feats, y, n_cls=100, folds=5, steps=400, wd=1e-3, seed=0):
    """Out-of-fold softmax-regression probabilities."""
    X = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
    N = len(X)
    perm = np.random.default_rng(seed).permutation(N)
    P = torch.zeros(N, n_cls, device=X.device)
    for f in range(folds):
        te = torch.as_tensor(perm[f::folds], device=X.device)
        tr_mask = torch.ones(N, dtype=torch.bool, device=X.device)
        tr_mask[te] = False
        W = torch.zeros(X.shape[1], n_cls, device=X.device, requires_grad=True)
        b = torch.zeros(n_cls, device=X.device, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=1e-2)
        for _ in range(steps):
            opt.zero_grad()
            loss = F.cross_entropy(X[tr_mask] @ W + b, y[tr_mask]) + wd * (W ** 2).sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            P[te] = F.softmax(X[te] @ W + b, -1)
    return P


def main():
    seed_all(0)
    model = load_orig()
    imgs = load_images_gpu()
    N = len(imgs)
    z16 = logits_of(model, imgs, fp16=True)
    z32 = logits_of(model, imgs, fp16=False)
    y = z16.argmax(1)
    p = F.softmax(z16, -1)
    top2 = p.topk(2, 1).values
    ce = ce_vec(z16, y)
    counts = torch.bincount(y, minlength=100).cpu().numpy()

    mild = torch.stack([logits_of(model, imgs, kind="mild", seed=100 + k) for k in range(N_MILD)])
    held = torch.stack([logits_of(model, imgs, kind="heldout", seed=900 + k) for k in range(4)])
    strong = torch.stack([logits_of(model, imgs, kind="strong", seed=500 + k) for k in range(N_STRONG)])
    s_aug = torch.stack([ce_vec(s, y) for s in strong]).mean(0)
    ce_mild = torch.stack([ce_vec(s, y) for s in mild])
    ce_held = torch.stack([ce_vec(s, y) for s in held])

    feats = mid_features(model, imgs)
    P_cv = probe_oof(feats, y)
    s_cv = -torch.log(P_cv.gather(1, y[:, None]).squeeze(1).clamp_min(1e-8))

    np.savez_compressed(
        ART / "orig_stats.npz",
        y=y.cpu().numpy(), z_clean=z16.cpu().numpy(), z_fp32=z32.cpu().numpy(),
        z_mild=mild.cpu().numpy().astype(np.float16), s_aug=s_aug.cpu().numpy(), s_cv=s_cv.cpu().numpy(),
        P_cv=P_cv.cpu().numpy(), counts=counts,
        p_strong=F.softmax(strong, -1).mean(0).cpu().numpy())

    def st(t):
        t = t.float().cpu().numpy()
        return {"mean": float(t.mean()), "std": float(t.std()), "p50": float(np.median(t)),
                "p90": float(np.quantile(t, .9)), "max": float(t.max())}

    res = {
        "N": N,
        "fp16_fp32_argmax_agree": float((z16.argmax(1) == z32.argmax(1)).float().mean()),
        "classes_present": int((counts > 0).sum()),
        "per_class_count": {"min": int(counts.min()), "max": int(counts.max()), "mean": float(counts.mean()),
                            "std": float(counts.std()), "top10": np.sort(counts)[::-1][:10].tolist()},
        "orig_conf": st(top2[:, 0]), "orig_margin": st(top2[:, 0] - top2[:, 1]),
        "frac_conf_lt_0.5": float((top2[:, 0] < 0.5).float().mean()),
        "orig_ce_clean": st(ce),
        "orig_ce_mild": st(ce_mild.mean(0)), "acc_mild": float((mild.argmax(-1) == y).float().mean()),
        "orig_ce_heldout": st(ce_held.mean(0)), "acc_heldout": float((held.argmax(-1) == y).float().mean()),
        "s_aug": st(s_aug), "acc_strong": float((strong.argmax(-1) == y).float().mean()),
        "s_cv": st(s_cv), "probe_oof_acc": float((P_cv.argmax(1) == y).float().mean()),
        "spearman_saug_scv": spearman(s_aug.cpu().numpy(), s_cv.cpu().numpy()),
        "spearman_saug_ceclean": spearman(s_aug.cpu().numpy(), ce.cpu().numpy()),
        "spearman_scv_ceclean": spearman(s_cv.cpu().numpy(), ce.cpu().numpy()),
    }
    dump_json(res, ART / "analysis.json")
    import json
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
