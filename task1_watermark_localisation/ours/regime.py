"""Step 1b: classify each image as mostly-watermarked (full=True) or mostly-clean from the global decoder output,
predict all-1 / all-0, report the official train score and write submissions/v1_regime.pt."""
import json
from pathlib import Path

import numpy as np
import torch

from common import CACHE, METHODS, meta, report

z = np.load(CACHE / "global_logits.npz", allow_pickle=True)
G = {n: v for n, v in zip(z["names"], z["vals"])}


def feats(name):
    s = G[name]
    return {"bitacc": float((s > 0).mean()), "mlogit": float(s.mean())}


train = meta("train")
thr = {}
for method in METHODS:
    xs = [x for x in train if x["method"] == method]
    for f in ("bitacc", "mlogit"):
        v = np.array([feats(x["image_name"])[f] for x in xs]); y = np.array([x["full_watermark"] for x in xs])
        print(f"{method:10s} {f:6s} full=True  min/med/max {v[y].min():.3f} {np.median(v[y]):.3f} {v[y].max():.3f}"
              f" | full=False min/med/max {v[~y].min():.3f} {np.median(v[~y]):.3f} {v[~y].max():.3f}")
    v = np.array([feats(x["image_name"])["bitacc"] for x in xs]); y = np.array([x["full_watermark"] for x in xs])
    cands = np.unique(v)
    accs = [((v >= t) == y).mean() for t in cands]
    best = int(np.argmax(accs))
    # middle of the best gap for robustness
    t = cands[best]; lower = cands[cands < t]
    thr[method] = float((t + lower.max()) / 2) if len(lower) else float(t)
    print(f"  -> threshold bitacc >= {thr[method]:.3f}  train acc {accs[best]:.3f}")


def predict(x):
    full = feats(x["image_name"])["bitacc"] >= thr[x["method"]]
    return np.full((128, 128), int(full), np.uint8)


report({x["image_name"]: predict(x) for x in train}, "v1 regime")
test = meta("test")
for method in METHODS:
    fr = np.mean([feats(x["image_name"])["bitacc"] >= thr[method] for x in test if x["method"] == method])
    print(f"test {method}: fraction predicted full = {fr:.3f}")
out = Path(__file__).parent / "submissions"; out.mkdir(exist_ok=True)
names = [x["image_name"] for x in test]
torch.save({"image_names": names, "masks": torch.from_numpy(np.stack([predict(x) for x in test])).bool()},
           out / "v1_regime.pt")
(CACHE / "regime_thresholds.json").write_text(json.dumps(thr))
print("wrote", out / "v1_regime.pt")
