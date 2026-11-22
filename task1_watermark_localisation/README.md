# Task 1: Watermark localization (public score 0.69)

Predict, for each of 1,020 test images, the 128×128 mask of the area watermarked by PixelSeal, TrustMark or
StegaStamp. The method and the embedded message of each test image are given.

- `ours/`: our code (one script per building block, versions v1 to v11).
- [`ours/METHODE_V8.md`](ours/METHODE_V8.md): **complete method description** (French). It covers the data facts
  that justify each block, every parameter, the evaluation protocol, the reproduction order with control numbers,
  and the list of ideas that did not work.
- `wmloc/`, `download_models.py`, `README.txt`, `requirements.txt`: the organisers' starter kit (model wrappers,
  official metric).

## Pipeline v11 (final)

| Step | What | Code |
|---|---|---|
| Regime | Global bit accuracy vs known message → "almost fully watermarked" or "partial" (thresholds 0.799 / 0.78 / 0.755) | [`regime.py`](ours/regime.py), [`global_logits.py`](ours/global_logits.py) |
| Agreement maps | Decoder per-position logits × known bits, shift-and-stitch (16 shifts) → 32×32 → 128×128 | [`maps.py`](ours/maps.py), [`feats.py`](ours/feats.py) |
| Synthetic data | COCO images and objects watermarked so that the statistics match the real data (`synth2`) | [`synth.py`](ours/synth.py) |
| U-Net | ResNet-34 U-Net + 3 auxiliary channels, 5 folds + full train (StegaStamp, 25 % of TrustMark) | [`train_seg.py`](ours/train_seg.py) |
| Probe | Segmentation head on the frozen official decoder's intermediate features (PixelSeal, TrustMark), plus a dilated head (v10) | [`ps_probe.py`](ours/ps_probe.py) |
| Decision | TTA (4 shifts), isotonic calibration per model, expected-mIoU decoding, component cleaning, guided filter | [`tta_infer.py`](ours/tta_infer.py), [`decision.py`](ours/decision.py), [`guided.py`](ours/guided.py) |
| Rectangles | Exact-crop search: decode candidate boxes, hill-climb, accept if bit acc ≥ 0.8 (PixelSeal, TrustMark) | [`global_rect.py`](ours/global_rect.py), [`rect_search.py`](ours/rect_search.py) |
| SAM | SAM ViT-B "everything" segments on predicted-partial images: greedy (PixelSeal), 0.5 blend (StegaStamp) | [`sam_masks.py`](ours/sam_masks.py), [`eval_v11.py`](ours/eval_v11.py) |
| Assembly | Writes the submission `{"image_names", "masks"}` | [`final_v11.py`](ours/final_v11.py) (v10: [`final_v10.py`](ours/final_v10.py)) |

Local estimates (out-of-fold, nested CV): v1 0.45, v6 0.62, v8 0.66, v9 0.68, v10 0.69, v11 ≈ 0.695.
Best public leaderboard score: 0.69.

## Reproduce

See §7 of [`ours/METHODE_V8.md`](ours/METHODE_V8.md) (25 ordered steps with expected control values).
Requirements: Python 3.12, torch 2.5.1, onnxruntime-gpu 1.23.2, trustmark 0.8.0, timm, segment-anything
(see `requirements.txt`). Data: Hugging Face dataset `SprintML/Hackathon_Watermark_Localisation_Task_Paris`.
Models: `python download_models.py`.
Note: the scripts write their heavy outputs to `~/wm_out/`, the home directory of the teammate who ran them.
