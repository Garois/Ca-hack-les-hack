"""StegaStamp: at the GT rectangle of real partial-rect images, compare bit accuracy of the exact (bicubic antialias)
crop path vs the roi_align path used by stage 1, and with small offsets."""
import numpy as np, torch
from torchvision.ops import roi_align
from common import bits, load_image, load_mask512, meta
from rect_search import Scorer
from global_rect import onnx_logits
sc = Scorer("stegastamp")
rows = []
for x in meta("train"):
    if x["method"] != "stegastamp" or x["full_watermark"]: continue
    g = load_mask512(x["image_name"]); ys, xs = np.where(g); b = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
    if g.sum() / ((b[1] - b[0]) * (b[3] - b[2])) < 0.95: continue
    img = torch.from_numpy(load_image("train", x["image_name"])).permute(2, 0, 1)[None].float().cuda() / 255
    sign = torch.from_numpy(bits(x["message"]).astype(np.float32) * 2 - 1).cuda()
    _, exact = sc(img, [b], sign)
    boxes = [b] + [(b[0] + d, b[1] + d, b[2] + d, b[3] + d) for d in (4, 8, 16)]
    boxes = [tuple(int(np.clip(v, 0, 512)) for v in bb) for bb in boxes]
    rois = torch.tensor([[0, bb[2], bb[0], bb[3], bb[1]] for bb in boxes], dtype=torch.float32, device="cuda")
    crops = roi_align(img, rois, output_size=(400, 400), sampling_ratio=0, aligned=True).clamp(0, 1)
    acc_roi = ((onnx_logits(sc.W, crops) * sign) > 0).float().mean(1).cpu().numpy()
    _, ex_off = sc(img, boxes[1:], sign)
    rows.append([exact[0], *acc_roi, *ex_off, b[1] - b[0], b[3] - b[2]])
r = np.array(rows)
print("n rects", len(r))
print("exact@GT mean %.3f  frac>=.8 %.2f" % (r[:, 0].mean(), (r[:, 0] >= .8).mean()))
print("roi@GT   mean %.3f  frac>=.8 %.2f" % (r[:, 1].mean(), (r[:, 1] >= .8).mean()))
for k, d in enumerate((4, 8, 16)):
    print(f"shift {d:2d}px: roi mean {r[:, 2 + k].mean():.3f} (>=.8 {(r[:, 2 + k] >= .8).mean():.2f}) | exact mean {r[:, 5 + k].mean():.3f} (>=.8 {(r[:, 5 + k] >= .8).mean():.2f})")
ok = r[:, 0] >= .8
print("sizes of decodable rects (h,w):", [(int(a), int(b)) for a, b in r[ok][:, -2:]])
