"""PixelSeal error budget on train (CPU only, cached data): current pipeline score per sub-type
(full/hole, partial/rect, partial/object), with the rect override where a search result exists."""
import json
import numpy as np
from common import CACHE, load_masks128, meta
from global_rect import OUT
from map_rules import is_full, rule
from rect_search import box_mask
from wmloc.utils import miou_image
gt = load_masks128(); R = json.loads((CACHE / "map_rules.json").read_text())
maps = dict(np.load(CACHE / "maps_pixelseal.npz"))
rects = {}
for f in list(CACHE.glob("rect_pixelseal_train*.json")) + list(OUT.glob("rect_pixelseal_val0*.json")):
    rects.update(json.loads(f.read_text()))
rows = {}
for x in meta("train"):
    if x["method"] != "pixelseal": continue
    n = x["image_name"]; g = gt[n]; full_p = is_full(x)
    small = g if not x["full_watermark"] else 1 - g
    ys, xs = np.where(small)
    shape = "rect" if len(ys) and small.sum() / ((ys.ptp() + 1) * (xs.ptp() + 1)) > 0.95 else "object"
    key = ("full" if x["full_watermark"] else "partial") + "/" + shape
    r = R[f"pixelseal/{full_p}"]; mk = rule(maps[n], full_p, r["q"], r["smooth"])
    s_rule = miou_image(mk, g)
    rr = rects.get(n)
    s = miou_image(box_mask(tuple(rr["box"])), g) if (rr and not full_p and rr["acc"] >= 0.8) else s_rule
    rows.setdefault(key, []).append((s_rule, s, rr is not None, g.mean(), full_p == x["full_watermark"]))
tot = sum(len(v) for v in rows.values())
for k, v in sorted(rows.items()):
    a = np.array(v, dtype=float)
    print(f"{k:15s} n={len(a):3d} ({len(a)/tot:.0%}) rule={a[:,0].mean():.3f} with-rect(where searched {int(a[:,2].sum())})={a[a[:,2]==1,1].mean() if a[:,2].any() else float('nan'):.3f} "
          f"wm-frac={a[:,3].mean():.2f} regime-ok={a[:,4].mean():.2f} | loss vs perfect = {(1-a[:,0]).sum()/tot:.3f}")
