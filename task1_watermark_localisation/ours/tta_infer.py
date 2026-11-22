"""PLAN_V9 B1 + B4: per-model probabilities (no TTA and 4-shift TTA) on each model's held-out fold (OOF) and on test.
Shift (dy, dx) in multiples of 4 px = whole cells of the 128 grid: the image content is moved by (dy, dx) (reflect
padding), auxiliary 128-maps by (dy/4, dx/4) cells, and the output is shifted back exactly.
    GPU=k run.sh tta_infer.py --kind unet  --method stegastamp
    GPU=k run.sh tta_infer.py --kind probe --method pixelseal
-> ~/wm_out/tta/<kind>_<method>_<model>_{oof,test}.npz with keys "<image>" (no TTA) and "<image>|tta"."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import bits, load_image, meta
from train_seg import N_AUX, Feats, UNet, ckpt_path, folds

OUT = Path.home() / "wm_out" / "tta"; OUT.mkdir(parents=True, exist_ok=True)
SHIFTS = [(0, 0), (4, 0), (0, 4), (4, 4)]


def shift_img(x, dy, dx, pad=32):                       # x_s[y, x] = x[y + dy, x + dx]
    if dy == 0 and dx == 0:
        return x
    xp = F.pad(x, (pad,) * 4, mode="reflect")
    return xp[..., pad + dy:pad + dy + x.shape[-2], pad + dx:pad + dx + x.shape[-1]]


def shift_cells(t, di, dj):                              # t_s[i, j] = t[i + di, j + dj] (replicate at borders)
    if di == 0 and dj == 0:
        return t
    tp = F.pad(t, (0, dj, 0, di), mode="replicate")
    return tp[..., di:di + t.shape[-2], dj:dj + t.shape[-1]]


def unshift_cells(o, di, dj):                            # inverse of shift_cells
    if di == 0 and dj == 0:
        return o
    op = F.pad(o, (dj, 0, di, 0), mode="replicate")
    return op[..., :o.shape[-2], :o.shape[-1]]


@torch.no_grad()
def run_model(fwd, items, split, need_aux=None, bs=16):
    """fwd(x_uint8_float[0,1] B x3x512x512, s Bxnbits, aux Bx3x128x128 or None) -> logits B x128x128."""
    out0, outt = {}, {}
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]
        x = torch.stack([torch.from_numpy(load_image(split, it["image_name"])).permute(2, 0, 1) for it in ch]).cuda().float() / 255
        s = torch.stack([torch.from_numpy(bits(it["message"]).astype(np.float32) * 2 - 1) for it in ch]).cuda()
        aux = torch.stack([need_aux(it["image_name"]) for it in ch]).cuda() if need_aux else None
        acc = 0
        for k, (dy, dx) in enumerate(SHIFTS):
            xs = shift_img(x, dy, dx); a = shift_cells(aux, dy // 4, dx // 4) if aux is not None else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = torch.sigmoid(fwd(xs, s, a).float())
            p = unshift_cells(p[:, None], dy // 4, dx // 4)[:, 0]
            if k == 0:
                p0 = p
            acc = acc + p
        pt = acc / len(SHIFTS)
        for it, a0, at in zip(ch, p0.cpu().numpy(), pt.cpu().numpy()):
            out0[it["image_name"]] = a0.astype(np.float16); outt[it["image_name"] + "|tta"] = at.astype(np.float16)
    return {**out0, **outt}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kind", required=True); ap.add_argument("--method", required=True)
    ap.add_argument("--utag", default="_s2", help="U-Net checkpoint suffix (models <m>_f<k><utag>)")
    ap.add_argument("--fam", default="fx", help="probe head family: heads <pre><fam><k> + <pre><allname>")
    ap.add_argument("--allname", default="all"); ap.add_argument("--dilated", action="store_true")
    ap.add_argument("--lmd", action="store_true")
    a = ap.parse_args(); m = a.method
    items, fold_of = folds(m); te = [x for x in meta("test") if x["method"] == m]
    t0 = time.time()
    if a.kind == "unet":
        feats = Feats(m, "feats2")
        models = [(f"{m}_f{k}{a.utag}", k) for k in range(5)] + [(f"{m}_f-1{a.utag}", None)]
        for name, k in models:
            net = UNet(pretrained=False, n_aux=N_AUX).cuda().eval()
            net.load_state_dict(torch.load(ckpt_path(name), map_location="cuda", weights_only=True))
            fwd = lambda x, s, aux: net(x, aux)
            if k is not None:
                held = [x for x in items if fold_of[x["image_name"]] == k]
                np.savez_compressed(OUT / f"unet_{m}_{name}_oof.npz", **run_model(fwd, held, "train", feats))
            np.savez_compressed(OUT / f"unet_{m}_{name}_test.npz", **run_model(fwd, te, "test", feats))
            print(name, f"{time.time() - t0:.0f}s", flush=True)
    else:
        from common import wm as load_wm
        from ps_probe import DIMS, OUT as PROBE, Backbone, BackboneTM, Head, HeadLMD
        W = load_wm(m, device="cuda"); bb = (Backbone(W) if m == "pixelseal" else BackboneTM(W)).cuda()
        pre = "" if m == "pixelseal" else "tm_"
        heads = [(f"{pre}{a.fam}{k}", k) for k in range(5)] + [(f"{pre}{a.allname}", None)]
        for tag, k in heads:
            head = (HeadLMD(DIMS[m], 256 if m == "pixelseal" else 100) if a.lmd else Head(True, DIMS[m], dilated=a.dilated)).cuda().eval()
            head.load_state_dict(torch.load(PROBE / f"head_{tag}.pt", weights_only=True))
            fwd = lambda x, s, aux: head(*bb(x, s), s=s)
            if k is not None:
                held = [x for x in items if fold_of[x["image_name"]] == k]
                np.savez_compressed(OUT / f"probe_{m}_{tag}_oof.npz", **run_model(fwd, held, "train"))
            np.savez_compressed(OUT / f"probe_{m}_{tag}_test.npz", **run_model(fwd, te, "test"))
            print(tag, f"{time.time() - t0:.0f}s", flush=True)
    print("done", m, a.kind)


if __name__ == "__main__":
    main()
