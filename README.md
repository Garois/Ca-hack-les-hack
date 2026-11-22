# Ca hack les hack: CISPA × Fraunhofer AISEC Hackathon, Paris 2026

Team **Ca hack les hack**. European AI Championship, Paris, 21–22 November 2026.
Three AI-security tasks, run on the JURECA supercomputer (A100 nodes).

| Task | Goal | Public score | Public rank | Code |
|---|---|---|---|---|
| 1. Watermark localization | Predict a 128×128 mask of the watermarked area (PixelSeal / TrustMark / StegaStamp) | **0.69** mIoU | 20th of 44 | [`task1_watermark_localisation/`](task1_watermark_localisation/) |
| 2. Model equality | Find which of 33 black-box LLM endpoints share the same weights / base family | **1.00** | **1st** (tied) | [`task2_model_equality/`](task2_model_equality/) |
| 3. Unlearning | Make a ViT-B/16 forget 3,500 images as if it had been retrained without them | **0.141** | 34th of 44 | [`task3_unlearning/`](task3_unlearning/) |
| **Overall** | | **0.64** | **20th of 44** | |

> Scores are on the public 30 % split, snapshot of 22 Nov 2026, 09:50 CET. The final ranking uses the hidden 70 %.
> **Slides:** [`slides/Ca-hack-les-hack_slides.pdf`](slides/Ca-hack-les-hack_slides.pdf) (speaker notes: [`slides/speaker_notes.md`](slides/speaker_notes.md))

## Team

| Member (JURECA login) | Main contribution |
|---|---|
| TODO name (`denoncin1`, team lead) | Team setup on JURECA, task kits and models |
| TODO name (`vaudois1`) | Task 1: watermark localization. Task 3: "local forgetting" explorations |
| TODO name (`darodesdetailly1`) | Task 2: model equality |
| TODO name (`lemasnedechermont1`) | Task 3: unlearning (best submission) |

---

## Task 1: Watermark localization (0.69)

**Problem.** There are 1,020 test images (512×512), each watermarked by **PixelSeal**, **TrustMark** or **StegaStamp**.
For each one we predict a 128×128 binary mask of the watermarked area. The metric is the per-image mean IoU over
the classes {0, 1}, averaged over images. There are two regimes: either the image is almost fully watermarked with
a "hole" (restored original or pasted object), or only a small rectangle or object is watermarked.

**Key idea.** At test time, the watermarking **method and the embedded message are known**. We use the official
decoders and the known message as *measuring instruments*:

1. **Regime.** The global bit accuracy of the decoded message separates "almost fully watermarked" from "partial"
   (97.6–98.8 % correct). This step alone scores 0.45 (all-zeros scores 0.25).
   → [`ours/regime.py`](task1_watermark_localisation/ours/regime.py), [`ours/global_logits.py`](task1_watermark_localisation/ours/global_logits.py)
2. **Probe on the frozen official decoder** (PixelSeal, TrustMark). A small segmentation head reads the decoder's
   intermediate features plus message-agreement maps (per-position logits × known bits, shift-and-stitch to 32×32).
   It is the main lever, because a pixel U-Net cannot "see" an imperceptible watermark.
   → [`ours/ps_probe.py#L27-L135`](task1_watermark_localisation/ours/ps_probe.py#L27-L135), [`ours/maps.py#L19-L45`](task1_watermark_localisation/ours/maps.py#L19-L45)
3. **U-Net ensemble** (StegaStamp, plus 25 % of TrustMark). The networks are trained on a *faithful* synthetic
   generator built from COCO, re-engineered until its statistics match the real data (objects keep the full-image
   watermark in place, rectangles are watermarked at their own scale).
   → [`ours/synth.py#L82-L123`](task1_watermark_localisation/ours/synth.py#L82-L123), [`ours/train_seg.py#L54-L95`](task1_watermark_localisation/ours/train_seg.py#L54-L95)
4. **Better decisions.** We use test-time augmentation (4 shifts), isotonic calibration per model, and a decoding
   rule that maximises the *expected* mIoU instead of using a fixed threshold, plus a guided filter.
   → [`ours/tta_infer.py`](task1_watermark_localisation/ours/tta_infer.py), [`ours/decision.py#L46-L90`](task1_watermark_localisation/ours/decision.py#L46-L90), [`ours/guided.py`](task1_watermark_localisation/ours/guided.py)
5. **Exact rectangle search.** Partial rectangles were watermarked as a crop, so decoding *exactly* the right box
   recovers the message. We run a coarse search, hill-climb the box, and accept it if bit accuracy ≥ 0.8.
   → [`ours/global_rect.py#L59-L97`](task1_watermark_localisation/ours/global_rect.py#L59-L97), [`ours/rect_search.py#L49-L103`](task1_watermark_localisation/ours/rect_search.py#L49-L103)
6. **SAM refinement** of partial objects (greedy segment selection for PixelSeal, 0.5 blend for StegaStamp).
   → [`ours/sam_masks.py`](task1_watermark_localisation/ours/sam_masks.py), [`ours/eval_v11.py`](task1_watermark_localisation/ours/eval_v11.py)
7. **Final assembly (v11)** → [`ours/final_v11.py`](task1_watermark_localisation/ours/final_v11.py)

**Validation.** Every change was adopted only if nested 5-fold cross-validation gave a paired bootstrap 95 % CI
with a lower bound above 0. The local estimate rose step by step: v1 0.45 → v6 0.62 → v8 0.66 → v9 0.68 →
v10 0.69 → v11 ≈ 0.695. Public leaderboard: 0.69.
Details, every parameter and the list of failed ideas (in French): [`ours/METHODE_V8.md`](task1_watermark_localisation/ours/METHODE_V8.md).

## Task 2: Model equality (1.00, tied 1st)

**Problem.** There are 33 black-box chat endpoints. For every pair we must say whether they share the same
**family** (base model, 10 % of the score) and the same **model** (identical weights, 90 %). Disguises such as a
system prompt, a persona, a watermark or temperature/top-p do *not* change the model. Quantization and LoRA *do*.
The budget is 100 answers per endpoint. The 9 free reference endpoints (`ref-01..09`, answers in
`practice_answers.csv`) served as our validation set.

**Key idea.** We identify each family's public base model, download it, and score **every token** the endpoints
return under that local model:

1. **Family.** The mean log-likelihood under candidate base models gives a clear winner: Qwen2.5-1.5B-Instruct,
   Qwen2.5-3B-Instruct or Llama-3.2-1B-Instruct, 11 endpoints each.
   → [`work/family_table.py`](task2_model_equality/work/family_table.py), [`work/fams.py`](task2_model_equality/work/fams.py)
2. **Decoding disguises.** We fit a maximum-likelihood temperature and top-p/top-k/min-p per endpoint. A base model
   served with top-p puts ~0 tokens outside its nucleus.
   → [`work/decfit.py`](task2_model_equality/work/decfit.py), [`work/decode_fit.py`](task2_model_equality/work/decode_fit.py), [`work/tboot.py`](task2_model_equality/work/tboot.py)
3. **Quantization.** A likelihood-ratio test compares the base model with a locally quantized copy (fp8, int8,
   int4, nf4, and the official AWQ/GPTQ checkpoints, de-quantized). Endpoints served quantized "prefer" the
   quantized copy.
   → [`work/llr.py#L24-L100`](task2_model_equality/work/llr.py#L24-L100), [`work/feat.py#L69-L104`](task2_model_equality/work/feat.py#L69-L104), [`work/extq.py`](task2_model_equality/work/extq.py)
4. **LoRA.** A learned-direction test fits a bias direction that explains one endpoint's token choices and projects
   every other endpoint onto it. Endpoints with the same adapter recognise each other, and base endpoints have
   α ≈ 0. Split-half reliability is cross-fitted.
   → [`work/ldt0.py`](task2_model_equality/work/ldt0.py), [`work/ldt.py`](task2_model_equality/work/ldt.py), [`work/ana2.py`](task2_model_equality/work/ana2.py), [`work/ana3.py`](task2_model_equality/work/ana3.py)
5. **Quiz prompts.** Two endpoints sharing an answer that is rare under the base model share a weight quirk.
   → [`work/qcal.py`](task2_model_equality/work/qcal.py), [`work/qpair.py`](task2_model_equality/work/qpair.py)
6. **Submission.** → [`work/write_sub.py`](task2_model_equality/work/write_sub.py). Final answer:
   [`work/subs/FINAL_public1.0.csv`](task2_model_equality/work/subs/FINAL_public1.0.csv). Each family has one base
   model behind 6–7 disguised endpoints, plus 4–5 distinct variants (fp8 / LoRA at different strengths).

Candidates built from the analysis alone scored 0.30–0.55 on the public split, and the remaining doubt sat in the
Qwen-3B family. Splitting out `ep-18` and then `ep-25` as their own models, after comparing a few alternative
partitions on the public split, gave 0.80 and then 1.00 (log: [`work/subs/log.jsonl`](task2_model_equality/work/subs/log.jsonl)).

## Task 3: Machine unlearning (0.141)

**Problem.** We get a ViT-B/16 (100 classes) and 3,500 unlabeled forget images. We must return weights that behave
like *gold* models retrained without those images: same loss distribution, same predicted distribution (including
*which* wrong class), also on hidden altered copies. At the same time the model must stay close to the original
elsewhere. Score = forget × retain, with a 70 % accuracy gate.

**Findings.**
- The original model is ~100 % confident on every forget image (mean CE 3·10⁻⁴), so they are memorised. 45 classes
  are present, and 9 of them have 129–130 images.
- "Whole classes were removed" was refuted: head-bias surgery on those 9 classes scored 0.
- The leaderboard kept rewarding **much stronger forgetting** than expected. As the target mean CE rose
  0.8 → 2.8 → 5.5 → 15, the score went 0.049 → 0.074 → 0.087 → 0.10+.

**Best pipeline** (`R26_null5f_e0.9_a2.65.pt`, public **0.1412**):
1. **Analysis.** Pseudo-labels are the original model's predictions. We compute logits on clean/augmented views and
   an out-of-fold linear probe on mid-layer features, which approximates "a model that never saw x" and gives a
   difficulty proxy.
   → [`code/analyze.py#L34-L80`](task3_unlearning/code/analyze.py#L34-L80)
2. **Soft targets q(x).** The true-class probability is p\* = sigmoid(a − b·u(x)), where u is the difficulty rank.
   The offset a is solved so that the mean CE equals μ = 15. The remaining mass follows the original model's own
   confusion distribution.
   → [`code/targets.py#L58-L99`](task3_unlearning/code/targets.py#L58-L99)
3. **Distillation.** Full fine-tuning for 3 epochs (AdamW, lr 1e-5) with KD toward q at T = 2 and T = 1. On
   augmented views the teacher is the original model's logits minus δ on the true class. We add KD to the original
   model on 6,000 public Imagenette photos as the retain proxy (weight 5).
   → [`code/unlearn.py#L90-L156`](task3_unlearning/code/unlearn.py#L90-L156)
4. **Null-space projection + extrapolation.** For each Linear layer we take the covariance of its input activations
   on the anchor photos. We remove from the weight change ΔW the top eigen-directions that hold 90 % of the energy
   (AlphaEdit/GPM style). This keeps 59.5 % of ‖Δ‖². We then push θ = θ₀ + 2.65·Δ_proj.
   → [`code/nullproj.py#L27-L98`](task3_unlearning/code/nullproj.py#L27-L98)
5. **Checks** (keys, dtypes, NaN, size) → [`code/validate_submission.py`](task3_unlearning/code/validate_submission.py),
   with a local proxy evaluator in [`code/evaluate_local.py`](task3_unlearning/code/evaluate_local.py).

The exact commands that produced the best file are in [`code/round13.slurm`](task3_unlearning/code/round13.slurm)
(training) and [`code/round15.slurm`](task3_unlearning/code/round15.slurm) (projection). The full run log is in
[`code/RESULTS.md`](task3_unlearning/code/RESULTS.md).

---

## Repository layout

```
task1_watermark_localisation/   ours/ = our code + METHODE_V8.md (full method, FR); wmloc/ = organisers' kit
task2_model_equality/           work/ = our code + README (FR); kit files at the top level; work/subs/ = submissions
task3_unlearning/               code/ = our pipeline + run scripts; artifacts/ = target configs;
                                exploration_vaudois1/ = second line of attack (local forgetting, literature notes)
slides/                         presentation
```

Not included because of size or confidentiality: datasets, model weights (the unlearning checkpoint is 328 MB),
API keys. Every script expects the organisers' task folders (`model.pt`, `forget_images.npz`,
`release/`, …) and was run on JURECA with the team's virtual environments. Paths are absolute JURECA paths and need
adapting. Each task folder has a README with reproduction steps.

All experiments ran on JURECA under the organisers' rules: one 4-GPU job per team and a low file count on the
shared scratch.
