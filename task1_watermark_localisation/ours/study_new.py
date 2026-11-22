"""New factors study (CPU, samples of real train images):
S1  TrustMark official REMOVER: residual d = remover(x) - x (at 256, upsampled). Maps: local energy |d| and local
    correlation of d with the expected TrustMark residual for the known message (re-embedding), pooled to 128.
S2  WMForger preference model (generic watermark-artifact detector, ConvNeXt): its global head applied per position
    at input 512 (16x16) and 768 (24x24) -> 'artifact' map. Sign convention checked from AUC.
S3  Train/test shift: global signed logit & bit accuracy of predicted-full images, train vs test.
S4  Hole error anatomy with the v9 decision on OOF TTA maps (missed / boundary / false holes)."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from common import CACHE, ROOT, bits, load_image, load_masks128, meta
from decision import apply_iso, emiou_mask, fit_iso, clean_full_or_partial
from exp_ps_maps import auc
from map_rules import is_full
from ps_probe import subtype
from train_seg import folds
from tta_eval import maps as oof_maps
from wmloc.utils import miou_image

gt = load_masks128(); torch.set_num_threads(16)
pool128 = lambda t: F.adaptive_avg_pool2d(t, 128)[0, 0].numpy()


def sample(m, k_full=10, k_obj=8):
    out = []
    for x in meta("train"):
        if x["method"] != m:
            continue
        st = subtype(x, gt[x["image_name"]])
        if (st == "full/object" and sum(1 for o in out if o[1] == st) < k_full) or (st == "partial/object" and sum(1 for o in out if o[1] == st) < k_obj):
            out.append((x, st))
    return out


def report(tag, rows):
    for st in ("full/object", "partial/object"):
        v = [a for s, a in rows if s == st]
        if v:
            print(f"   {tag:34s} {st:15s} n={len(v):2d} AUC {np.nanmean(v):.3f}", flush=True)


# ---------------- S1: TrustMark remover
from wmloc.watermarks import load_watermark
TM = load_watermark("trustmark", ROOT / "models", device="cpu"); tm = TM.tm
rows_e, rows_c, rows_n = [], [], []
for x, st in sample("trustmark"):
    n = x["image_name"]; img = load_image("train", n); g = gt[n]
    st256 = torch.from_numpy(np.asarray(__import__("PIL.Image", fromlist=["Image"]).fromarray(img).resize((256, 256), 2))).permute(2, 0, 1)[None].float() / 127.5 - 1
    with torch.no_grad():
        d = (tm.removal(st256).clamp(-1, 1) - st256)                       # 1x3x256x256, ~ -watermark
    d512 = F.interpolate(d, (512, 512), mode="bilinear", align_corners=False)
    e = pool128(d512.abs().mean(1, keepdim=True)); rows_e.append((st, auc(e.ravel(), g.ravel())))
    # expected residual for the known message (float re-embedding through the encoder at 512)
    w = np.asarray(TM.embed(img, bits(x["message"])), np.float32) / 127.5 - 1 - (img.astype(np.float32) / 127.5 - 1)
    r = torch.from_numpy(w).permute(2, 0, 1)[None]
    num = F.avg_pool2d((-d512 * r).sum(1, keepdim=True), 16, 4, padding=6)
    den = torch.sqrt(F.avg_pool2d((d512 ** 2).sum(1, keepdim=True), 16, 4, padding=6) * F.avg_pool2d((r ** 2).sum(1, keepdim=True), 16, 4, padding=6) + 1e-12)
    c = F.interpolate(num / den, (128, 128), mode="bilinear", align_corners=False)[0, 0].numpy()
    rows_c.append((st, auc(c.ravel(), g.ravel())))
print("== S1 TrustMark remover")
report("energy |remover(x)-x|", rows_e); report("corr(-d, expected residual)", rows_c)

# ---------------- S2: WMForger per-position artifact map
sys.path.insert(0, str(ROOT / "models" / "videoseal" / "wmforger"))
import omegaconf
from wmforger.models import build_extractor
cfg = omegaconf.OmegaConf.load(str(ROOT / "models" / "videoseal" / "wmforger" / "configs" / "extractor.yaml"))["convnext_tiny"]
wf = build_extractor("convnext_tiny", cfg, img_size=256, nbits=0)
wf.load_state_dict(torch.load(Path.home() / "wm_out" / "wmforger" / "convnext_pref_model.pth", weights_only=True, map_location="cpu")["model"]); wf.eval()


@torch.no_grad()
def wf_map(img, size):
    x = F.interpolate(torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255, (size, size), mode="bilinear", align_corners=False, antialias=True)
    x = wf.preprocess(x); lat = wf.convnext(x); emb = wf.pixel_decoder.output_upscaling(lat); lin = wf.pixel_decoder.linear
    pp = torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None]
    return F.interpolate(pp[:, :1], (128, 128), mode="bilinear", align_corners=False)[0, 0].numpy(), float(wf(x * 0.5 + 0.5).mean()) if False else None


print("== S2 WMForger per-position map (AUC; < 0.5 means the sign is inverted)")
for m in ("pixelseal", "trustmark", "stegastamp"):
    for size in (512, 768):
        rows = []
        for x, st in sample(m, 8, 6):
            mp, _ = wf_map(load_image("train", x["image_name"]), size); rows.append((st, auc(mp.ravel(), gt[x["image_name"]].ravel())))
        report(f"{m} WMForger@{size}", rows)

# ---------------- S3: train/test shift on global statistics (predicted-full images)
print("== S3 train vs test global statistics (predicted-full images): median mean-signed-logit / bit acc")
for m in ("pixelseal", "trustmark", "stegastamp"):
    z = np.load(CACHE / f"feats2_{m}.npz"); idx = {n: i for i, n in enumerate(z["names"].tolist())}; glob = z["glob"]
    for split in ("train", "test"):
        v = np.array([glob[idx[x["image_name"]]] for x in meta(split) if x["method"] == m and is_full(x)])
        print(f"   {m:10s} {split:5s} n={len(v):3d} mean signed logit median {np.median(v[:, 0]):.2f} | bit acc median {np.median(v[:, 1]):.3f}")

# ---------------- S4: hole error anatomy (v9 decision, OOF TTA maps)
print("== S4 hole anatomy (GT full images, v9 decision)")
for m in ("pixelseal", "trustmark", "stegastamp"):
    items, fold_of = folds(m); P = oof_maps(m, True); names = [x["image_name"] for x in items]; X = {x["image_name"]: x for x in items}
    fullp = {n: is_full(X[n]) for n in names}; Q = {}
    for f in range(5):
        for full in (True, False):
            tr = [n for n in names if fold_of[n] != f and fullp[n] == full]
            cal = fit_iso(np.concatenate([P[n].ravel() for n in tr]), np.concatenate([gt[n].ravel() for n in tr]).astype(float))
            for n in names:
                if fold_of[n] == f and fullp[n] == full:
                    Q[n] = apply_iso(cal, P[n])
    stats = []
    for n in names:
        if not X[n]["full_watermark"]:
            continue
        g = gt[n]; mk = clean_full_or_partial(emiou_mask(ndimage.gaussian_filter(Q[n], 1)), 16)
        hg, hp = (g == 0), (mk == 0)
        inter = (hg & hp).sum(); iou0 = inter / max((hg | hp).sum(), 1)
        stats.append((hg.mean(), hp.mean(), iou0, inter / max(hg.sum(), 1), inter / max(hp.sum(), 1) if hp.sum() else np.nan))
    s = np.array(stats, dtype=float)
    print(f"   {m:10s} n={len(s)} hole IoU mean {s[:, 2].mean():.2f} | missed (recall<0.2) {np.mean(s[:, 3] < 0.2):.2f} | "
          f"predicted no hole {np.mean(s[:, 1] == 0):.2f} | GT hole frac {s[:, 0].mean():.3f} vs predicted {s[:, 1].mean():.3f} | "
          f"precision {np.nanmean(s[:, 4]):.2f} recall {s[:, 3].mean():.2f}")
