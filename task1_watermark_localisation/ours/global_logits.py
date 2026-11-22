"""Step 1: global decoder logits for every train/test image -> cache/global_logits.npz
Stores per image the signed logits s*logit (s = 2*msg-1), so >0 means the bit agrees with the known message."""
import numpy as np
import torch

from common import CACHE, METHODS, bits, load_image, meta, wm

out = {}
for split in ("train", "test"):
    for method in METHODS:
        items = [x for x in meta(split) if x["method"] == method]
        w = wm(method)
        for i in range(0, len(items), 32):
            chunk = items[i:i + 32]
            imgs = torch.stack([torch.from_numpy(load_image(split, x["image_name"])).permute(2, 0, 1).float() / 255
                                for x in chunk])
            lg = w.logits(imgs.cuda() if method != "stegastamp" else imgs).cpu().numpy()
            for x, l in zip(chunk, lg):
                s = bits(x["message"]).astype(np.float32) * 2 - 1
                out[x["image_name"]] = s * l
        print(split, method, len(items), flush=True)
np.savez(CACHE / "global_logits.npz", names=np.array(list(out)), vals=np.array(list(out.values()), dtype=object),
         allow_pickle=True)
print("done")
