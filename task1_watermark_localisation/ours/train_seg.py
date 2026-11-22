"""Step 3: U-Net segmentation (128x128 output) trained on synthetic data + real train, one model per method.
    run.sh train_seg.py --method pixelseal --fold 0 [--aux]   # real train split in 5 folds, fold 0 held out
    run.sh train_seg.py --method pixelseal --fold -1 [--aux]  # final model: all real train used for training
Input: RGB + high-pass residual (6 channels) at 512x512; with --aux also the message-aware features of feats.py
(128x128 message map + 2 global regime scalars), injected in every decoder block and in the head.
Output: logits at 128x128 (stride 4), the official grid.
Loss: BCE + (1 - soft per-image mIoU over both classes), which mirrors the leaderboard metric."""
import argparse
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common import CACHE, ROOT, load_image, load_masks128, meta
from wmloc.utils import miou_image

CKPT = ROOT / "ours" / "ckpt"; CKPT.mkdir(parents=True, exist_ok=True)
CKPT_HOME = Path.home() / "wm_out" / "ckpt"          # fallback: the shared scratch quota is intermittently full


def save_ckpt(obj, fname, saver=torch.save):
    """Save to ours/ckpt; on quota errors fall back to ~/wm_out/ckpt (and remove a stale scratch copy)."""
    try:
        saver(obj, CKPT / fname); return CKPT / fname
    except OSError as e:
        print("WARNING scratch save failed, using home:", e, flush=True)
        CKPT_HOME.mkdir(parents=True, exist_ok=True)
        try:
            (CKPT / fname).unlink(missing_ok=True)
        except OSError:
            pass
        saver(obj, CKPT_HOME / fname); return CKPT_HOME / fname


def ckpt_path(name):
    """Most recent of ours/ckpt/<name>.pt and ~/wm_out/ckpt/<name>.pt."""
    c = [p for p in (CKPT / f"{name}.pt", CKPT_HOME / f"{name}.pt") if p.exists()]
    return max(c, key=lambda p: p.stat().st_mtime)
N_AUX = 3


def highpass(x):
    """x: B x 3 x H x W in [0,1] -> high-frequency residual (where the watermarks live), amplified."""
    blur = F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), 5, stride=1)
    return (x - blur) * 10


class UNet(nn.Module):
    def __init__(self, encoder="resnet34.a1_in1k", pretrained=True, n_aux=0):
        super().__init__()
        self.n_aux = n_aux
        self.enc = timm.create_model(encoder, pretrained=pretrained, features_only=True, in_chans=6)
        ch = self.enc.feature_info.channels()                      # strides 2, 4, 8, 16, 32
        self.stem = nn.Sequential(nn.Conv2d(6, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 32, 3, stride=2, padding=1),
                                  nn.GELU(), nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.GELU())  # full-res path -> s4
        dec = []
        cin = ch[4]
        for skip in (ch[3], ch[2], ch[1]):                          # up to stride 4
            dec.append(nn.Sequential(nn.Conv2d(cin + skip + n_aux, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
                                     nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU()))
            cin = 128
        self.dec = nn.ModuleList(dec)
        self.head = nn.Sequential(nn.Conv2d(128 + 32 + n_aux, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 1, 1))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, aux=None):                                  # x: B x 3 x 512 x 512 in [0,1]; aux: B x n_aux x 128 x 128
        inp = torch.cat([(x - self.mean) / self.std, highpass(x)], 1)
        f = self.enc(inp)
        y = f[4]
        for blk, skip in zip(self.dec, (f[3], f[2], f[1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            parts = [y, skip]
            if self.n_aux:
                parts.append(F.interpolate(aux, size=skip.shape[-2:], mode="bilinear", align_corners=False))
            y = blk(torch.cat(parts, 1))
        parts = [y, self.stem(inp)] + ([aux] if self.n_aux else [])
        return self.head(torch.cat(parts, 1))[:, 0]                  # B x 128 x 128 logits


def soft_miou_loss(logits, target):
    p = torch.sigmoid(logits).flatten(1); g = target.flatten(1)
    s = 0
    for pc, gc in ((p, g), (1 - p, 1 - g)):
        inter = (pc * gc).sum(1); union = (pc + gc - pc * gc).sum(1)
        s = s + (inter + 1) / (union + 1)
    return 1 - (s / 2).mean()


class Feats:
    """name -> aux tensor (3 x 128 x 128): message map / 5, mean signed logit / 5, bit accuracy * 2 - 1."""

    def __init__(self, method, name="feats"):
        z = np.load(CACHE / f"{name}_{method}.npz")
        self.idx = {n: i for i, n in enumerate(z["names"].tolist())}
        self.maps = z["maps"]; self.glob = z["glob"]

    def __contains__(self, name):
        return name in self.idx

    def __call__(self, name):
        i = self.idx[name]
        m = torch.from_numpy(self.maps[i].astype(np.float32)) / 5
        g0 = torch.full_like(m, float(self.glob[i, 0]) / 5); g1 = torch.full_like(m, float(self.glob[i, 1]) * 2 - 1)
        return torch.stack([m, g0, g1])


class SynthDS(Dataset):
    """Synthetic samples. Old sets (synth/) are one png+npz per sample; new sets (synth2/) are shards (synth.py)."""

    def __init__(self, method, feats=None, synth_dir="synth"):
        d = ROOT / "ours" / synth_dir / method
        self.items = []                                              # (name, shard idx or png path, index, mask128)
        self.shards = []
        if any(d.glob("*_meta.npz")):
            from synth import iter_shards
            for k, (imgs, m) in enumerate(iter_shards(method, synth_dir)):
                self.shards.append(imgs.filename)
                for j, n in enumerate(m["names"].tolist()):
                    self.items.append((n, k, j, m["mask128"][j]))
        else:
            for f in sorted(d.glob("*.npz")):
                self.items.append((f.stem, f.with_suffix(".png"), -1, None))
        self.feats = feats
        if feats is not None:
            self.items = [it for it in self.items if it[0] in feats]
        self._mm = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        n, src, j, m = self.items[i]
        if j >= 0:
            if src not in self._mm:                                  # one memmap per shard and worker
                self._mm[src] = np.load(self.shards[src], mmap_mode="r")
            img = np.array(self._mm[src][j])
        else:
            img = np.asarray(Image.open(src).convert("RGB")); m = np.load(src.with_suffix(".npz"))["mask128"]
        aux = self.feats(n) if self.feats is not None else torch.zeros(0)
        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(m.astype(np.float32)), aux


class RealDS(Dataset):
    def __init__(self, items, feats=None):
        self.items = items; self.gt = load_masks128(); self.feats = feats

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x = self.items[i]; n = x["image_name"]
        img = load_image("train", n)
        aux = self.feats(n) if self.feats is not None else torch.zeros(0)
        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(self.gt[n].astype(np.float32)), aux


def folds(method, k=5):
    """Deterministic stratified folds over (full_watermark) for one method."""
    items = sorted([x for x in meta("train") if x["method"] == method], key=lambda x: x["image_name"])
    out = {}
    for full in (False, True):
        grp = [x for x in items if x["full_watermark"] == full]
        for j, x in enumerate(grp):
            out[x["image_name"]] = j % k
    return items, out


@torch.no_grad()
def predict_probs(model, imgs_uint8, aux=None, bs=16):
    """imgs: N x 3 x 512 x 512 uint8 tensor (aux: N x 3 x 128 x 128) -> N x 128 x 128 probabilities."""
    model.eval(); out = []
    for i in range(0, len(imgs_uint8), bs):
        x = imgs_uint8[i:i + bs].cuda().float() / 255
        a = aux[i:i + bs].cuda() if aux is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(model(x, a).float())
        out.append(p.cpu())
    return torch.cat(out)


def load_split(split, items, feats=None):
    imgs = torch.stack([torch.from_numpy(load_image(split, x["image_name"])).permute(2, 0, 1) for x in items])
    aux = torch.stack([feats(x["image_name"]) for x in items]) if feats is not None else None
    return imgs, aux


def evaluate(model, items, gt, feats=None):
    imgs, aux = load_split("train", items, feats)
    probs = predict_probs(model, imgs, aux).numpy()
    res = {}
    for thr in (0.3, 0.4, 0.5, 0.6):
        s = [miou_image((p > thr).astype(np.uint8), gt[x["image_name"]]) for p, x in zip(probs, items)]
        res[thr] = round(float(np.mean(s)), 4)
    by = {}
    for p, x in zip(probs, items):
        by.setdefault(x["full_watermark"], []).append(miou_image((p > 0.5).astype(np.uint8), gt[x["image_name"]]))
    return res, {k: round(float(np.mean(v)), 4) for k, v in by.items()}, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--encoder", default="resnet34.a1_in1k")
    ap.add_argument("--real_rep", type=int, default=4, help="how many times real train images are repeated per epoch")
    ap.add_argument("--aux", action="store_true"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--synth_dir", default="synth"); ap.add_argument("--feats", default="feats")
    ap.add_argument("--init", default="", help="checkpoint name to start from (fine-tuning)")
    ap.add_argument("--tag", default=""); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    feats = Feats(a.method, a.feats) if a.aux else None
    items, fold_of = folds(a.method)
    tr_real = [x for x in items if fold_of[x["image_name"]] != a.fold]
    va_real = [x for x in items if fold_of[x["image_name"]] == a.fold]
    gt = load_masks128()
    synth = [SynthDS(a.method, feats, a.synth_dir)] if a.synth_dir != "none" else []
    ds = torch.utils.data.ConcatDataset(synth + [RealDS(tr_real, feats)] * a.real_rep)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=True)
    model = UNet(a.encoder, n_aux=N_AUX if a.aux else 0).cuda()
    if a.init:
        model.load_state_dict(torch.load(ckpt_path(a.init), map_location="cuda", weights_only=True))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    name = f"{a.method}_f{a.fold}{a.tag}"
    print(f"{name}: synth={len(synth[0]) if synth else 0} real_train={len(tr_real)} real_val={len(va_real)} steps={steps}", flush=True)
    best = -1
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0
        for x, y, aux in dl:
            x = x.cuda(non_blocking=True).float() / 255; y = y.cuda(non_blocking=True)
            aux = aux.cuda(non_blocking=True) if a.aux else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(x, aux).float()
            loss = F.binary_cross_entropy_with_logits(lg, y) + soft_miou_loss(lg, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        msg = f"ep {ep} loss {tot / len(dl):.4f} {time.time() - t0:.0f}s"
        if va_real:
            res, by, probs = evaluate(model, va_real, gt, feats)
            msg += f" | val {res} by_full {by}"
            if res[0.5] > best:
                best = res[0.5]; save_ckpt(model.state_dict(), f"{name}.pt")
                save_ckpt(None, f"{name}_valprobs.npz", saver=lambda _, p: np.savez_compressed(
                    p, names=np.array([x["image_name"] for x in va_real]), probs=probs.astype(np.float16)))
        else:
            save_ckpt(model.state_dict(), f"{name}.pt")
        print(msg, flush=True)
    print("best", best)


if __name__ == "__main__":
    main()
