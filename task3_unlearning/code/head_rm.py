"""R1: head-bias surgery for presumed fully-removed classes (forget count >= --cap).
head.bias[c] -= D shifts logit c by exactly -D, so all stats are computed analytically from cached orig logits.
D is solved so that the median clean CE over the removed images equals each --ce target."""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from common import ANCHORS, ART, HERE, OUT, ce_vec, kl_vec, load_images_gpu, load_orig, logits_of, save_state_dict

ap = argparse.ArgumentParser()
ap.add_argument("--cap", type=int, default=129)
ap.add_argument("--ce", default="5,10")
ap.add_argument("--tag", default="R1")
args = ap.parse_args()

S = np.load(ART / "orig_stats.npz")
counts = S["counts"]
y = torch.as_tensor(S["y"]).long()
cls = np.where(counts >= args.cap)[0]
rm = torch.as_tensor(counts[S["y"]] >= args.cap)
print("removed classes:", cls.tolist(), "n_img:", int(rm.sum()))

orig = load_orig()
imgs = load_images_gpu()
z = {"clean": torch.as_tensor(S["z_clean"]).float(),
     "mild": torch.as_tensor(S["z_mild"][0]).float(),
     "heldout": logits_of(orig, imgs, kind="heldout", seed=4242).cpu()}
anc = np.load(ANCHORS)["val"]
z["anchor"] = logits_of(orig, torch.as_tensor(anc).cuda()).cpu()
shift = torch.zeros(100)
shift[torch.as_tensor(cls)] = 1.0


def st(t):
    return {"mean": round(float(t.mean()), 3), "std": round(float(t.std()), 3), "med": round(float(t.median()), 3)}


def solve(target):
    lo, hi = 0.0, 60.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if ce_vec(z["clean"] - mid * shift, y)[rm].median() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


forget_cls = torch.as_tensor(counts > 0)
report = {}
for tgt in [float(s) for s in args.ce.split(",")]:
    D = solve(tgt)
    r = {"D": round(D, 3)}
    for k in ("clean", "mild", "heldout"):
        zs = z[k] - D * shift
        ce = ce_vec(zs, y)
        pred = zs.argmax(1)
        r[k] = {"all": st(ce), "removed": st(ce[rm]), "kept": st(ce[~rm]),
                "acc_removed": round(float((pred[rm] == y[rm]).float().mean()), 3),
                "removed_pred_in_forget_classes": round(float(forget_cls[pred[rm]].float().mean()), 3),
                "kl_orig_sub_removed_T2": round(float(kl_vec(z[k][rm], zs[rm]).mean()), 3)}
    za = z["anchor"] - D * shift
    r["anchor"] = {"agree": round(float((za.argmax(1) == z["anchor"].argmax(1)).float().mean()), 4),
                   "kl_T2": round(float(kl_vec(z["anchor"], za).mean()), 4)}
    # rough control-set estimate: 9 of 100 classes behave like the held-out views of removed images
    frac = len(cls) / 100
    k_rm = r["heldout"]["kl_orig_sub_removed_T2"]
    r["est_retain"] = round((1 - frac) / (1 + frac * k_rm + (1 - frac) * r["anchor"]["kl_T2"]), 3)
    sd = {k: v.clone() for k, v in orig.state_dict().items()}
    sd["head.bias"][torch.as_tensor(cls, device=sd["head.bias"].device)] -= D
    d = OUT / args.tag
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"headrm_ce{tgt:g}.pt"
    save_state_dict(sd, p)
    r["path"] = str(p)
    report[f"ce{tgt:g}"] = r
    print(json.dumps(r, indent=1), flush=True)
(OUT / args.tag / "report.json").write_text(json.dumps(report, indent=2))
