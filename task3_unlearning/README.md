# Task 3: Machine unlearning (public score 0.1412)

We get a ViT-B/16 (100 classes, `model.pt`) and 3,500 unlabeled forget images (`forget_images.npz`). We must
produce weights that behave like *gold* models retrained without those images, both on them and on hidden altered
copies, while staying close to the original model elsewhere.
Score = forget (loss mean/std + KL to the gold predictive distribution) × retain (accuracy ratio × 1/(1+KL) on a
hidden control set), with a 70 % accuracy gate.

- `code/`: our pipeline, plus the SLURM rounds that produced every candidate (`round*.slurm`). Every run appends
  its local metrics to [`code/RESULTS.md`](code/RESULTS.md) (earlier runs: [`code/RESULTS_early.md`](code/RESULTS_early.md)).
- `artifacts/`: configuration and statistics of every soft-target set we built. `analysis.json` holds the forget-set
  statistics of the original model.
- `exploration_vaudois1/`: our second line of attack. It covers local, layer-restricted forgetting, SalUn masks,
  the "9 classes removed" probe, a literature review and the submission journal.
- `task_template.py`: the organisers' template (model definition, preprocessing, scoring description).

## Best submission: `R26_null5f_e0.9_a2.65.pt` (sha256 `fede5622a9a976f6…`, 328 MB, not in git)

```bash
export TASK_DIR=<folder with task_template.py, model.pt, forget_images.npz> UNL_WORK=<output folder>
export UNL_ANCHORS=<anchors.npz>        # built by code/prepare_anchors.py from Imagenette-320 (6,000 train + 1,500 val photos)
python code/analyze.py                  # pseudo-labels, augmented-view logits, out-of-fold probe -> artifacts/orig_stats.npz
python code/targets.py --mu 15.0 --b 1.5 --rest orig --class_mode sample --floor 1e-7 --tag or_mu150
python code/unlearn.py --targets $UNL_WORK/artifacts/targets_or_mu150.npz --epochs 3 --w_anchor 5.0 --alphas 1.0 --tag R24_mu150anc5
python code/nullproj.py --src $UNL_WORK/output/R24_mu150anc5/alpha_1.pt --targets $UNL_WORK/artifacts/targets_or_mu150.npz \
       --eps 0.9 --alphas 2.65 --tag R26_null5f
python code/validate_submission.py $UNL_WORK/output/R26_null5f/eps0.9_alpha_2.65.pt
```

| Step | Idea | Code |
|---|---|---|
| Analysis | The original model is ~100 % confident on all forget images (mean CE 3·10⁻⁴); 45 classes present. An out-of-fold linear probe on blocks 6–9 approximates "a model that never saw x" | [`analyze.py`](code/analyze.py) |
| Soft targets | p\*(x) = sigmoid(a − b·u(x)), with u = normal quantile of the probe-difficulty rank and a solved so that mean CE = μ (= 15). The non-true mass follows the original model's confusion distribution; confidence is never raised | [`targets.py`](code/targets.py) |
| Distillation | Full fine-tuning, 3 epochs, AdamW lr 1e-5 (cosine), KD at T=2 (+0.5 at T=1) toward q. On mild views the teacher is orig(view) − δ·e_y. Retain proxy: KD to orig on Imagenette photos (×5) | [`unlearn.py`](code/unlearn.py) |
| Null-space projection | Per Linear layer: covariance of input activations on anchor photos; remove from ΔW its component on the top eigenvectors covering 90 % of the energy (AlphaEdit / GPM). This keeps 59.5 % of ‖Δ‖² | [`nullproj.py`](code/nullproj.py) |
| Extrapolation | θ = θ₀ + α·Δ_proj, α = 2.65 | [`nullproj.py`](code/nullproj.py), [`extrap.py`](code/extrap.py) |
| Local proxy scoring | Targets stand in for gold. The retain proxy is fold-out images, mixes and anchor drift | [`evaluate_local.py`](code/evaluate_local.py) |

Local metrics of the best file: forget CE mean/std 2.08 / 2.51 (accuracy 0.50), held-out altered views 2.56 / 2.83,
anchor agreement with the original 0.56 (KL 0.53).

## Public-score history (what the leaderboard taught us)

| Submission | Change | Public |
|---|---|---|
| R1 | Head-bias surgery: remove the 9 "full" classes | 0 (refutes whole-class removal) |
| P1b | Soft targets, μ = 0.8 | 0.049 |
| R9 → R13 | μ = 1.2 → 1.6 → 2.2 → 2.8 → 3.6 | 0.054 → 0.062 → 0.069 → 0.074 → 0.078 |
| R15 | Non-true mass = orig confusion, μ = 5.5 | 0.087 |
| R19 | μ = 15 + anchors, extrapolated ×1.2 | 0.1035 |
| R23 / R24 | + null-space projection (ε = 0.9) of the anchor ×2 / ×3 runs, α = 1.8 / 2.0 | 0.1342 / 0.1367 |
| **R26_null5f** | anchors ×5, ε = 0.9, **α = 2.65** | **0.1412** |
| R25 / R25b / R27 / R28 | α = 2.8 / ε = 0.95 / soup of 2 seeds / ε = 0.85 | 0.1399 / 0.1408 / 0.1383 / 0.1364 |

Did not help: sharpened targets (0.052), anchors at moderate μ (0.065, 0.079), local forgetting restricted to a
few layers (0.055), weight soups (0.138).
