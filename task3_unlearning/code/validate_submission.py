"""Validate a candidate state dict against the submission rules. Writes <file>.validated with sha256 on success."""
import hashlib
import sys
from pathlib import Path

import torch

from common import DEVICE, MODEL_PATH, load_images_gpu, logits_of, tt


def main(path):
    path = Path(path)
    errs = []
    size_mb = path.stat().st_size / 2 ** 20
    if path.suffix not in (".pt", ".pth"):
        errs.append("suffix")
    if size_mb > 450:
        errs.append(f"size {size_mb:.1f}MB")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    ref = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    if set(sd) != set(ref):
        errs.append(f"keys differ: extra={sorted(set(sd) - set(ref))[:5]} missing={sorted(set(ref) - set(sd))[:5]}")
    for k, v in sd.items():
        if k.startswith("module.") or k.startswith("_orig_mod."):
            errs.append(f"prefix {k}")
            break
        if k in ref and v.shape != ref[k].shape:
            errs.append(f"shape {k}")
        if v.is_floating_point():
            if v.dtype != torch.float32:
                errs.append(f"dtype {k} {v.dtype}")
            if not torch.isfinite(v).all():
                errs.append(f"nonfinite {k}")
    m = tt.build_model()
    m.load_state_dict(sd, strict=True)
    m = m.to(DEVICE).eval()
    imgs = load_images_gpu()[:256]
    z = logits_of(m, imgs)
    if not torch.isfinite(z).all():
        errs.append("fp16 forward non-finite")
    diff = sum(float((sd[k].float() - ref[k].float()).abs().max()) > 0 for k in ref if k in sd)
    if errs:
        print("INVALID:", errs)
        return 1
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(str(path) + ".validated").write_text(h + "\n")
    print(f"VALID {path} size={size_mb:.1f}MB tensors_changed={diff}/{len(ref)} sha256={h}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
