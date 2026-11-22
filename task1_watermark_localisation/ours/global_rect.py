"""Global search of the own-scale watermarked rectangle (partial regime, ~half of partial images are such rectangles).
Stage 1: thousands of candidate boxes (multi-size, multi-aspect, strided) are cropped+resized in one GPU op
         (roi_align) and decoded; score = mean signed logit against the known message.
Stage 2: the best few (after NMS) are hill-climbed with the exact antialiased crop (rect_search.hill_climb).
Accept the rectangle if its final bit accuracy >= ACC.
    GPU=3 run.sh global_rect.py --method trustmark --split train      # evaluation on real partial train images
    GPU=3 run.sh global_rect.py --method trustmark --split test       # cache results for test images
-> cache/rect_<method>_<split>.json : name -> {"box": [y0,y1,x0,x1], "acc": float, "score": float}"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_iou, roi_align

from common import CACHE, bits, load_image, load_masks128, meta
from rect_search import RES, Scorer, box_mask, hill_climb
from wmloc.utils import miou_image

MAX_RATIO = {"pixelseal": 16, "stegastamp": 16, "trustmark": 2}
SIZES = [28, 40, 56, 80, 112, 160, 224, 320, 448, 512]
OUT = Path(os.environ.get("WM_OUT", str(Path.home() / "wm_out"))); OUT.mkdir(parents=True, exist_ok=True)
BATCH = {"pixelseal": 384, "trustmark": 1024, "stegastamp": 64}
STRIDE_DIV = {"pixelseal": 3, "trustmark": 3, "stegastamp": 2}   # stegastamp decoder is ~25x slower


def candidates(method):
    out = []
    for h in SIZES:
        for w in SIZES:
            if max(h, w) / min(h, w) > MAX_RATIO[method] + 1e-6:
                continue
            sy, sx = max(8, h // STRIDE_DIV[method]), max(8, w // STRIDE_DIV[method])
            for y in list(range(0, 512 - h + 1, sy)) + [512 - h]:
                for x in list(range(0, 512 - w + 1, sx)) + [512 - w]:
                    out.append((y, y + h, x, x + w))
    return np.array(sorted(set(out)), dtype=np.float32)


def onnx_logits(W, crops):
    """StegaStamp decoder on GPU crops without going through the CPU: ONNX Runtime IO binding on torch buffers.
    crops: B x 3 x 400 x 400 float cuda in [0,1] (already resized) -> B x 100 logits (cuda)."""
    x = crops.permute(0, 2, 3, 1).contiguous().float()
    sec = torch.zeros(len(x), 100, device=x.device)
    out = torch.empty(len(x), 100, device=x.device)
    io = W.sess.io_binding()
    io.bind_input("image", "cuda", 0, np.float32, tuple(x.shape), x.data_ptr())
    io.bind_input("secret", "cuda", 0, np.float32, tuple(sec.shape), sec.data_ptr())
    io.bind_output(W.LOGITS, "cuda", 0, np.float32, tuple(out.shape), out.data_ptr())
    torch.cuda.synchronize()
    W.sess.run_with_iobinding(io)
    return out


@torch.no_grad()
def stage1(sc, img_t, sign, cands):
    res, mode = RES[sc.method]; bs = BATCH[sc.method]
    scores = []
    for i in range(0, len(cands), bs):
        c = torch.from_numpy(cands[i:i + bs]).cuda()
        rois = torch.cat([torch.zeros(len(c), 1, device=c.device), c[:, [2, 0, 3, 1]]], 1)   # (idx, x0, y0, x1, y1)
        crops = roi_align(img_t, rois, output_size=(res, res), spatial_scale=1.0, sampling_ratio=0, aligned=True)
        if mode == "bicubic":
            crops = crops.clamp(0, 1)
        if sc.method == "stegastamp":
            lg = onnx_logits(sc.W, crops) * sign
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = sc.W.logits(crops).float() * sign
        scores.append(lg.mean(1))
    return torch.cat(scores)


def search(sc, img, msg, cands, topk=6):
    img_t = torch.from_numpy(img).permute(2, 0, 1)[None].float().cuda() / 255
    sign = torch.from_numpy(msg.astype(np.float32) * 2 - 1).cuda()
    s = stage1(sc, img_t, sign, cands)
    order = torch.argsort(s, descending=True)[:200]
    boxes = torch.from_numpy(cands).cuda()[order][:, [2, 0, 3, 1]]                 # xyxy for NMS
    keep = []
    for i in range(len(order)):
        if all(box_iou(boxes[i:i + 1], boxes[j:j + 1]).item() < 0.5 for j in keep):
            keep.append(i)
        if len(keep) == topk:
            break
    best = (None, -1e9, 0.0)
    for i in keep:
        b = tuple(int(v) for v in cands[order[i].item()])
        box, sc_, acc = hill_climb(sc, img_t, sign, b)
        if sc_ > best[1]:
            best = (box, sc_, acc)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--split", default="train")
    ap.add_argument("--names", default="", help="json list of image names to process (default: all of the split)")
    ap.add_argument("--n", type=int, default=0); ap.add_argument("--out", default="")
    a = ap.parse_args()
    sc = Scorer(a.method); cands = candidates(a.method)
    print(a.method, "candidates per image:", len(cands), flush=True)
    items = [x for x in meta(a.split) if x["method"] == a.method]
    if a.split == "train":
        items = [x for x in items if not x["full_watermark"]]
    if a.names:
        keep = set(json.loads(open(a.names).read())); items = [x for x in items if x["image_name"] in keep]
    if a.n:
        items = items[:a.n]
    # Results go to $HOME (separate quota): the shared scratch quota of the project is intermittently full.
    # Resume: names already present in the output (home or old cache copy) are skipped.
    fname = a.out or f"rect_{a.method}_{a.split}.json"
    out_path = OUT / fname
    out = {}
    for p in (CACHE / fname, out_path):
        if p.exists():
            out.update(json.loads(p.read_text()))
    todo = [x for x in items if x["image_name"] not in out]
    print(f"{len(out)} already done, {len(todo)} to do -> {out_path}", flush=True)
    t0 = time.time()
    for k, x in enumerate(todo):
        box, s, acc = search(sc, load_image(a.split, x["image_name"]), bits(x["message"]), cands)
        out[x["image_name"]] = {"box": list(box), "acc": acc, "score": s}
        if k % 10 == 0:
            print(f"{k}/{len(todo)} {time.time() - t0:.0f}s acc={acc:.2f}", flush=True)
        if k % 5 == 0 or k == len(todo) - 1:                                          # checkpoint partial results
            try:
                tmp = out_path.with_suffix(".tmp"); tmp.write_text(json.dumps(out)); tmp.rename(out_path)
            except OSError as e:
                print("WARNING could not write results:", e, flush=True)
    if a.split == "train":
        gt = load_masks128()
        for thr in (0.7, 0.75, 0.8, 0.85, 0.9):
            rows = []
            for n, r in out.items():
                g = gt[n]; ys, xs = np.where(g); is_rect = g.sum() / ((ys.ptp() + 1) * (xs.ptp() + 1)) > 0.95
                hit = r["acc"] >= thr
                rows.append((is_rect, hit, miou_image(box_mask(r["box"]), g) if hit else np.nan))
            r = np.array(rows, dtype=float)
            print(f"acc>={thr}: fires on {np.mean(r[r[:, 0] == 1, 1]):.2f} of rects, {np.mean(r[r[:, 0] == 0, 1]):.2f} of objects;"
                  f" mIoU when fired: rects {np.nanmean(r[(r[:, 0] == 1) & (r[:, 1] == 1), 2]):.3f}"
                  f" objects {np.nanmean(r[(r[:, 0] == 0) & (r[:, 1] == 1), 2]) if ((r[:, 0] == 0) & (r[:, 1] == 1)).any() else float('nan'):.3f}")
    print("time", f"{time.time() - t0:.0f}s for {len(items)} images")


if __name__ == "__main__":
    main()
