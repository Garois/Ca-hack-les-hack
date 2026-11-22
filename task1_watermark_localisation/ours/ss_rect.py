"""B4 (GPU): seeded rectangle search for StegaStamp (PLAN_V10).
The decoder needs the rectangle CENTRE within ~2 px (translation of 4 px -> acc 0.60) but tolerates single-edge moves
(4 px -> 0.84). So: centre on a 4 px grid within +-24 px of the seed centre x size = seed size x {0.6,0.8,1,1.25,1.6}
per axis (~4 200 boxes, one roi_align + ONNX pass), NMS, then exact-path hill-climb (4, 2, 1 px) of the best 3.
-> ~/wm_out/ssrect_<split>[_<shard>].json {name: {box, acc, score, seed}} (resumable)."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_iou

from common import bits, load_image, meta
from global_rect import stage1
from rect_search import Scorer, clip_box, hill_climb

OUT = Path.home() / "wm_out"
FACT = (0.6, 0.8, 1.0, 1.25, 1.6)


def candidates(seed):
    y0, y1, x0, x1 = seed; cy, cx = (y0 + y1) / 2, (x0 + x1) / 2; h, w = y1 - y0, x1 - x0
    out = set()
    for dy in range(-24, 25, 4):
        for dx in range(-24, 25, 4):
            for fh in FACT:
                for fw in FACT:
                    H, Wd = max(16, h * fh), max(16, w * fw)
                    b = clip_box((cy + dy - H / 2, cy + dy + H / 2, cx + dx - Wd / 2, cx + dx + Wd / 2))
                    out.add(b)
    return np.array(sorted(out), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="train")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()
    seeds = json.loads((OUT / "ss_seeds.json").read_text())[a.split]
    msg = {x["image_name"]: x["message"] for x in meta(a.split) if x["method"] == "stegastamp"}
    names = sorted(seeds)[a.shard::a.nshard]
    fout = OUT / (f"ssrect_{a.split}.json" if a.nshard == 1 else f"ssrect_{a.split}_{a.shard}.json")
    res = json.loads(fout.read_text()) if fout.exists() else {}
    sc = Scorer("stegastamp"); t0 = time.time()
    todo = [n for n in names if n not in res]
    print(f"{len(res)} done, {len(todo)} to do", flush=True)
    for k, n in enumerate(todo):
        img_t = torch.from_numpy(load_image(a.split, n)).permute(2, 0, 1)[None].float().cuda() / 255
        sign = torch.from_numpy(bits(msg[n]).astype(np.float32) * 2 - 1).cuda()
        cands = candidates(seeds[n]); s = stage1(sc, img_t, sign, cands)
        order = torch.argsort(s, descending=True)[:100]; boxes = torch.from_numpy(cands).cuda()[order][:, [2, 0, 3, 1]]
        keep = []
        for i in range(len(order)):
            if all(box_iou(boxes[i:i + 1], boxes[j:j + 1]).item() < 0.5 for j in keep):
                keep.append(i)
            if len(keep) == 3:
                break
        best = (None, -1e9, 0.0)
        for i in keep:
            b, scr, acc = hill_climb(sc, img_t, sign, tuple(int(v) for v in cands[order[i].item()]), steps=(4, 2, 1), max_iter=10)
            if scr > best[1]:
                best = (b, scr, acc)
        res[n] = {"box": list(best[0]), "acc": best[2], "score": best[1], "seed": seeds[n]}
        if k % 5 == 0 or k == len(todo) - 1:
            tmp = fout.with_suffix(".tmp"); tmp.write_text(json.dumps(res)); tmp.rename(fout)
            print(f"{k + 1}/{len(todo)} {time.time() - t0:.0f}s acc={best[2]:.2f} cands={len(cands)}", flush=True)
    print("done", a.split, len(res))


if __name__ == "__main__":
    main()
