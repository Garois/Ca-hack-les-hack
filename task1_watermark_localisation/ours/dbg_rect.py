import sys, numpy as np, torch
from common import ROOT, bits, load_image, load_mask512, load_masks128, meta
from rect_search import Scorer, init_boxes, hill_climb, clip_box
from wmloc.utils import miou_image
m = sys.argv[1]; sc = Scorer(m); gt = load_masks128()
z = np.load(ROOT / "ours" / f"ckpt/{m}_f0_aux_valprobs.npz"); P = dict(zip(z["names"].tolist(), z["probs"].astype(np.float32)))
M = {x["image_name"]: x for x in meta("train")}
for n, p in P.items():
    x = M[n]
    if x["full_watermark"]: continue
    g = load_mask512(n); ys, xs = np.where(g); gb = (ys.min(), ys.max()+1, xs.min(), xs.max()+1); fill = g.sum()/((gb[1]-gb[0])*(gb[3]-gb[2]))
    img = load_image("train", n); msg = bits(x["message"])
    it = torch.from_numpy(img).permute(2,0,1)[None].float().cuda()/255; sg = torch.from_numpy(msg.astype(np.float32)*2-1).cuda()
    gs, ga = sc(it, [clip_box(gb)], sg)
    boxes = init_boxes(p)
    res = [hill_climb(sc, it, sg, b) for b in boxes]
    best = max(res, key=lambda r: r[1]) if res else None
    print(f"{n} rect={fill>0.95} gtfrac={g.mean():.3f} predfrac={(p>0.5).mean():.3f} pmax={p.max():.2f} nbox={len(boxes)} GTbox acc={ga[0]:.2f} | best acc={best[2] if best else -1:.2f} box={best[0] if best else None} gt={tuple(int(v) for v in gb)}")
