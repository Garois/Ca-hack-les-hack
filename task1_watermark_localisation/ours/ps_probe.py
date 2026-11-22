"""PixelSeal local detector: frozen PixelSeal ConvNeXt (it has learned to see the PixelSeal pattern) + a small trained
segmentation head on its intermediate stages (strides 8/16/32 px at 512) and on the known-message agreement map.
    GPU=0 run.sh ps_probe.py --tag base                 # synth2 only; ALL real train images are held out
    GPU=0 run.sh ps_probe.py --tag f0 --real_fold 0     # + real train except fold 0 (OOF for fold 0)
Prints, per epoch, AUC and mIoU@0.5 per real sub-type; saves the head and the probabilities on held-out real images
(and on test with --predict_test) to ~/wm_out/ps_probe/."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from common import bits, load_image, load_masks128, meta
from common import wm as load_wm
from exp_ps_maps import auc
from synth import iter_shards
from train_seg import folds, soft_miou_loss
from wmloc.utils import miou_image

OUT = Path.home() / "wm_out" / "ps_probe"; OUT.mkdir(parents=True, exist_ok=True)


class Backbone(nn.Module):
    """Frozen PixelSeal detector -> stage-1..3 features, per-position message agreement and detection channel."""

    def __init__(self, W):
        super().__init__(); self.W = W; self.det = W.det.eval()
        for p in self.det.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x512, s):
        x = self.det.preprocess(self.W._resize(x512)); cn = self.det.convnext; f = []
        for i in range(4):
            x = cn.downsample_layers[i](x); x = cn.stages[i](x); f.append(x)
        emb = self.det.pixel_decoder.output_upscaling(f[3]); lin = self.det.pixel_decoder.linear
        pp = torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None]
        agree = (pp[:, 1:] * s[:, :, None, None]).mean(1, keepdim=True)
        return f[0], f[1], f[2], agree, pp[:, :1]


class BackboneTM(nn.Module):
    """Frozen TrustMark decoder (ResNet-50 at 245) -> layer1..3 features and per-position message agreement."""

    def __init__(self, W):
        super().__init__(); self.W = W; self.net = W.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x512, s):
        n = self.net; x = self.W._prep(x512)
        x = n.maxpool(n.relu(n.bn1(n.conv1(x)))); f = []
        for k in ("layer1", "layer2", "layer3", "layer4"):
            x = getattr(n, k)(x); f.append(x)
        pp = torch.einsum("bchw,oc->bohw", f[3], n.fc.weight) + n.fc.bias[None, :, None, None]
        agree = (pp * s[:, :, None, None]).mean(1, keepdim=True)
        return f[0], f[1], f[2], agree, torch.zeros_like(agree)


class BackboneSS(nn.Module):
    """Frozen StegaStamp decoder (ONNX graph run by ONNX Runtime on GPU with IO binding, see onnx_feats.py)
    -> dec_conv_4/5/6 activations (64@100, 64@50, 128@25 at 400 input) and a 13x13 per-position message agreement
    obtained by linearising the two dense layers around their operating point (exact decomposition of the
    pre-activations, ReLU mask taken from the actual forward)."""
    NAMES = ["gen_decoder/dec_conv_4/Relu:0", "gen_decoder/dec_conv_5/Relu:0", "gen_decoder/dec_conv_6/Relu:0",
             "gen_decoder/dec_conv_7/Relu:0", "gen_decoder/dec_dense/BiasAdd:0", "gen_decoder/decoder_out/BiasAdd:0"]
    SHAPES = [(64, 100, 100), (64, 50, 50), (128, 25, 25), (128, 13, 13), (512,), (100,)]

    def __init__(self, W=None):
        super().__init__()
        import onnxruntime as ort
        d = Path.home() / "wm_out" / "stega"
        so = ort.SessionOptions(); so.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(str(d / "stega_feats.onnx"), so,
                                         providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])
        z = np.load(d / "stega_dense.npz")
        self.register_buffer("W1", torch.from_numpy(z["W1"]).float().view(13, 13, 128, 512))   # NHWC flatten order
        self.register_buffer("W2", torch.from_numpy(z["W2"]).float())

    @torch.no_grad()
    def forward(self, x512, s):
        B = x512.shape[0]
        x = F.interpolate(x512.float(), size=(400, 400), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        img = x.permute(0, 2, 3, 1).contiguous(); sec = torch.zeros(B, 100, device=x.device)
        outs = [torch.empty((B,) + sh, device=x.device) for sh in self.SHAPES]
        io = self.sess.io_binding()
        io.bind_input("image", "cuda", 0, np.float32, tuple(img.shape), img.data_ptr())
        io.bind_input("secret", "cuda", 0, np.float32, tuple(sec.shape), sec.data_ptr())
        for n, t in zip(self.NAMES, outs):
            io.bind_output(n, "cuda", 0, np.float32, tuple(t.shape), t.data_ptr())
        torch.cuda.synchronize(); self.sess.run_with_iobinding(io); io.synchronize_outputs(); torch.cuda.synchronize()
        c4, c5, c6, c7, pre, _ = outs
        u = torch.einsum("bchw,hwcj->bhwj", c7, self.W1)                # per-position share of the dense pre-activations
        ws = torch.einsum("jk,bk->bj", self.W2, s) * (pre > 0).float()  # d(agreement)/d(pre-activation), ReLU mask
        agree = torch.einsum("bhwj,bj->bhw", u, ws)[:, None] / 100
        return c4, c5, c6, agree, torch.zeros_like(agree)


DIMS = {"pixelseal": (96, 192, 384), "trustmark": (256, 512, 1024), "stegastamp": (64, 64, 128)}


def proj(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 1), nn.BatchNorm2d(cout))


class Head(nn.Module):
    """dilated=True (v10): the 4 conv3x3 blocks use dilations 1, 2, 4, 8 -> receptive field ~31 cells of the f1 grid
    (~250 px at 512) instead of ~9 cells (~70 px), so the inside of large holes/objects is seen in context.
    extra (v11): number of additional input maps given at the f1 resolution (e.g. local message agreement)."""

    def __init__(self, use_f1=True, dims=(96, 192, 384), dilated=False, extra=0):
        super().__init__(); self.use_f1 = use_f1
        self.p1 = proj(dims[0], 64); self.p2 = proj(dims[1], 64); self.p3 = proj(dims[2], 64)
        cin = 64 * (3 if use_f1 else 2) + 3 + extra
        dl = (1, 2, 4, 8) if dilated else (1, 1, 1, 1)
        blk = lambda i, o, d: nn.Sequential(nn.Conv2d(i, o, 3, padding=d, dilation=d), nn.BatchNorm2d(o), nn.GELU())
        self.body = nn.Sequential(blk(cin, 128, dl[0]), blk(128, 128, dl[1]), blk(128, 128, dl[2]), blk(128, 64, dl[3]),
                                  nn.Conv2d(64, 1, 1))

    def forward(self, f1, f2, f3, agree, det0, s=None, extra=None):
        size = f1.shape[-2:]
        up = lambda t: F.interpolate(t, size=size, mode="bilinear", align_corners=False)
        g = agree.mean((2, 3), keepdim=True).expand(-1, -1, *size)
        parts = ([self.p1(f1)] if self.use_f1 else []) + [up(self.p2(f2)), up(self.p3(f3)), up(agree) / 5, g / 5, up(det0) / 5]
        if extra:
            parts += [up(e) for e in extra]
        y = self.body(torch.cat(parts, 1))
        return F.interpolate(y, size=(128, 128), mode="bilinear", align_corners=False)[:, 0]


class HeadLMD(nn.Module):
    """v11 'local message decoder': a branch predicts the nbits message logits at EVERY cell of the f1 grid (64x64,
    8 px at 512) from stages 1-3; its agreement with the known message (mean_k s_k L_k) and its confidence
    (mean_k |L_k|) are fed as 2 extra maps to the dilated segmentation head. Trained jointly: segmentation loss +
    bit BCE on watermarked cells + penalty pushing the agreement to 0 on clean cells."""

    def __init__(self, dims, nbits):
        super().__init__()
        self.seg = Head(True, dims, dilated=True, extra=2)
        self.b1 = proj(dims[0], 128); self.b2 = proj(dims[1], 128); self.b3 = proj(dims[2], 128)
        self.bits = nn.Sequential(nn.GELU(), nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
                                  nn.Conv2d(128, nbits, 1))

    def forward(self, f1, f2, f3, agree, det0, s=None, return_bits=False):
        size = f1.shape[-2:]
        up = lambda t: F.interpolate(t, size=size, mode="bilinear", align_corners=False)
        L = self.bits(self.b1(f1) + up(self.b2(f2)) + up(self.b3(f3)))
        a_loc = (L * s[:, :, None, None]).mean(1, keepdim=True); e_loc = L.abs().mean(1, keepdim=True)
        y = self.seg(f1, f2, f3, agree, det0, extra=[a_loc, e_loc])
        return (y, L, a_loc) if return_bits else y


def lmd_losses(L, a_loc, y128, s):
    """bit BCE on watermarked cells (weighted by the watermarked fraction of each 2x2 block of the 128 grid) and a
    squared-agreement penalty on clean cells."""
    w = F.adaptive_avg_pool2d(y128[:, None], L.shape[-2:])                     # B x1 x h x w in [0,1]
    tgt = ((s + 1) / 2)[:, :, None, None].expand_as(L)
    bce = F.binary_cross_entropy_with_logits(L, tgt, reduction="none").mean(1, keepdim=True)
    l_bits = (bce * w).sum() / (w.sum() + 1e-6)
    l_clean = ((a_loc ** 2) * (1 - w)).sum() / ((1 - w).sum() + 1e-6)
    return l_bits, l_clean


class SynthPS(Dataset):
    def __init__(self, method="pixelseal"):
        self.items = []; self.files = []
        for k, (imgs, m) in enumerate(iter_shards(method, "synth2")):
            self.files.append(imgs.filename)
            for j in range(len(m["names"])):
                self.items.append((k, j, m["mask128"][j], str(m["message"][j])))
        self._mm = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        k, j, m, msg = self.items[i]
        if k not in self._mm:
            self._mm[k] = np.load(self.files[k], mmap_mode="r")
        img = np.array(self._mm[k][j])
        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(m.astype(np.float32)), torch.from_numpy(bits(msg).astype(np.float32) * 2 - 1)


class RealPS(Dataset):
    def __init__(self, items, split="train"):
        self.items = items; self.split = split; self.gt = load_masks128() if split == "train" else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x = self.items[i]; img = load_image(self.split, x["image_name"])
        m = self.gt[x["image_name"]].astype(np.float32) if self.gt is not None else np.zeros((128, 128), np.float32)
        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(m), torch.from_numpy(bits(x["message"]).astype(np.float32) * 2 - 1)


def subtype(x, g):
    small = g if not x["full_watermark"] else 1 - g; ys, xs = np.where(small)
    shape = "rect" if len(ys) and small.sum() / ((ys.ptp() + 1) * (xs.ptp() + 1)) > 0.95 else "object"
    return ("full" if x["full_watermark"] else "partial") + "/" + shape


@torch.no_grad()
def predict(bb, head, items, split):
    head.eval(); out = {}
    dl = DataLoader(RealPS(items, split), batch_size=32, num_workers=8)
    k = 0
    for x, _, s in dl:
        x = x.cuda().float() / 255; s = s.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(head(*bb(x, s), s=s).float())
        for pp in p.cpu().numpy():
            out[items[k]["image_name"]] = pp.astype(np.float16); k += 1
    head.train(); return out


def evaluate(probs, items):
    gt = load_masks128(); rows = {}
    for x in items:
        n = x["image_name"]; g = gt[n]; p = probs[n].astype(np.float32)
        rows.setdefault(subtype(x, g), []).append((auc(p.ravel(), g.ravel()), miou_image((p > 0.5).astype(np.uint8), g)))
    allm = np.mean([r[1] for v in rows.values() for r in v])
    return allm, {k: (round(float(np.nanmean([r[0] for r in v])), 3), round(float(np.mean([r[1] for r in v])), 4), len(v)) for k, v in sorted(rows.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base"); ap.add_argument("--real_fold", type=int, default=-2,
                                                               help="-2: no real data in training; k: real train except fold k; -1: all real")
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--real_rep", type=int, default=8); ap.add_argument("--no_f1", action="store_true")
    ap.add_argument("--predict_test", action="store_true")
    ap.add_argument("--fixed", action="store_true", help="keep the LAST epoch (no selection on the held-out fold)")
    ap.add_argument("--method", default="pixelseal")
    ap.add_argument("--dilated", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lmd", action="store_true", help="v11 local message decoder head (implies dilated)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if a.method == "stegastamp":
        bb = BackboneSS().cuda()                    # own ONNX session with exposed intermediate outputs
    else:
        W = load_wm(a.method, device="cuda")
        bb = (Backbone(W) if a.method == "pixelseal" else BackboneTM(W)).cuda()
    head = (HeadLMD(DIMS[a.method], {"pixelseal": 256, "trustmark": 100, "stegastamp": 100}[a.method]) if a.lmd
            else Head(not a.no_f1, DIMS[a.method], dilated=a.dilated)).cuda()
    items, fold_of = folds(a.method)
    if a.real_fold == -2:
        tr_real, held = [], items
    elif a.real_fold == -1:
        tr_real, held = items, []
    else:
        tr_real = [x for x in items if fold_of[x["image_name"]] != a.real_fold]; held = [x for x in items if fold_of[x["image_name"]] == a.real_fold]
    ds = ConcatDataset([SynthPS(a.method)] + ([RealPS(tr_real)] * a.real_rep if tr_real else []))
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=12, drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl), pct_start=0.1)
    print(f"[{a.tag}] train samples {len(ds)} (real {len(tr_real)} x{a.real_rep}), held-out real {len(held)}", flush=True)
    best = -1
    for ep in range(a.epochs):
        t0 = time.time(); tot = 0
        for x, y, s in dl:
            x = x.cuda(non_blocking=True).float() / 255; y = y.cuda(non_blocking=True); s = s.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if a.lmd:
                    lg, L, a_loc = head(*bb(x, s), s=s, return_bits=True)
                else:
                    lg = head(*bb(x, s), s=s)
            lg = lg.float()
            loss = F.binary_cross_entropy_with_logits(lg, y) + soft_miou_loss(lg, y)
            if a.lmd:
                l_bits, l_clean = lmd_losses(L.float(), a_loc.float(), y, s)
                loss = loss + 0.5 * l_bits + 0.1 * l_clean
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        msg = f"[{a.tag}] ep {ep} loss {tot / len(dl):.4f} {time.time() - t0:.0f}s"
        if held:
            probs = predict(bb, head, held, "train"); allm, per = evaluate(probs, held)
            msg += f" | held-out mIoU@.5 {allm:.4f} " + " ".join(f"{k}:AUC{v[0]}/mIoU{v[1]}(n{v[2]})" for k, v in per.items())
            if allm > best or (a.fixed and ep == a.epochs - 1):
                best = allm; torch.save(head.state_dict(), OUT / f"head_{a.tag}.pt")
                np.savez_compressed(OUT / f"heldprobs_{a.tag}.npz", **probs)
        else:
            torch.save(head.state_dict(), OUT / f"head_{a.tag}.pt")
        print(msg, flush=True)
    if a.predict_test:
        head.load_state_dict(torch.load(OUT / f"head_{a.tag}.pt", weights_only=True))
        te = [x for x in meta("test") if x["method"] == a.method]
        np.savez_compressed(OUT / f"testprobs_{a.tag}.npz", **predict(bb, head, te, "test"))
        print("test probabilities saved", flush=True)
    print(f"[{a.tag}] best held-out mIoU {best:.4f}")


if __name__ == "__main__":
    main()
