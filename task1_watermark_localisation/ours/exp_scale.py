"""On REAL partial images: decode crops around the GT region bbox expanded by factor s (clipped to the image).
If the watermark was embedded on a rectangle of a given scale, the crop matching that rectangle decodes best."""
import numpy as np
import torch

from common import METHODS, bits, load_image, load_mask512, meta
from common import wm as load_wm

S = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 99]
import sys
method = sys.argv[1]
W = load_wm(method)
xs = [x for x in meta("train") if x["method"] == method and not x["full_watermark"]]
accs = []
for x in xs:
    img = load_image("train", x["image_name"]); m = load_mask512(x["image_name"]); msg = bits(x["message"])
    ys, xx = np.where(m); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xx.min(), xx.max() + 1
    cy, cx, h, w = (y0 + y1) / 2, (x0 + x1) / 2, y1 - y0, x1 - x0
    crops = []
    for s in S:
        H, Wd = min(512, h * s), min(512, w * s)
        a0 = int(np.clip(cy - H / 2, 0, 512 - H)); b0 = int(np.clip(cx - Wd / 2, 0, 512 - Wd))
        crops.append(img[a0:a0 + int(H), b0:b0 + int(Wd)])
    row = []
    for c in crops:
        t = torch.from_numpy(np.ascontiguousarray(c)).permute(2, 0, 1)[None].float() / 255
        row.append(float((W.decode(t).cpu().numpy()[0] == msg).mean()))
    accs.append(row)
A = np.array(accs)
fill = []
print(method, "mean acc per expansion:", {s: round(float(v), 3) for s, v in zip(S, A.mean(0))})
print(method, "fraction of images with max acc >= 0.9:", float((A.max(1) >= 0.9).mean()),
      " argmax distribution:", {s: int(c) for s, c in zip(S, np.bincount(A.argmax(1), minlength=len(S)))})
