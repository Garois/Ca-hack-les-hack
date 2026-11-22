#!/bin/bash
#SBATCH --job-name=unl_round10
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --exclude=jrc[0198,0215,0217,0231-0233,0245,0257-0258,0283,0314,0321,0323-0324,0327,0329,0347,0349-0350,0362-0363,0374,0387,0394,0396,0410,0414,0423,0442,0444]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=logs/round10-%j.out
#SBATCH --error=logs/round10-%j.out
# ONE job, ONE unreserved node (reserved nodes excluded), 4 GPUs in parallel. Run from a teammate account:
#   cd "/p/scratch/training2666/denoncin1/Ca hack les hack/unlearning/lemasnedechermont1" && sbatch code/round10.slurm
SCR="/p/scratch/training2666/denoncin1/Ca hack les hack/unlearning"
W="$SCR/lemasnedechermont1"
export TASK_DIR="$SCR" UNL_WORK="$W" UNL_ANCHORS="$W/anchors/anchors.npz" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
source "$SCR/.venv/bin/activate"
C="$W/code"; O="$W/output"; A="$W/artifacts"; SUB="$W/submit"; L="$W/logs"
fin() { python "$C/validate_submission.py" "$1" && cp "$1" "$SUB/$2" && echo "READY $2 $(sha256sum "$SUB/$2" | cut -c1-16)"; }
# G0: NegMerge (sign-consensus) of 8 anchored original-shape models, then push
( export CUDA_VISIBLE_DEVICES=0
  python "$C/negmerge.py" --targets "$A/targets_or_mu150.npz" --alphas 1.0,1.3,1.6,1.9 --tag R21_negm \
    --srcs "$SUB/R15_or_mu55anc_a1.pt,$SUB/R17_or_mu80anc_a1.pt,$SUB/R18_or_mu80anc_s1_a1.pt,$SUB/R18_or_mu80anc1_a1.pt,$SUB/R18_or_mu110anc_a1.pt,$SUB/R18_or_mu110anc_s1_a1.pt,$SUB/R18_or_mu150anc_a1.pt,$SUB/R18_or_mu150anc1_a1.pt" \
    | grep -E "merged|^alpha|Error"
  for a in 1 1.3 1.6 1.9; do fin "$O/R21_negm/alpha_$a.pt" R21_negm_a$a.pt; done ) > "$L/r10_g0.log" 2>&1 &
# G1: null-space projection of the best anchored task vector (protects directions used by the anchor photos)
( export CUDA_VISIBLE_DEVICES=1
  python "$C/nullproj.py" --src "$SUB/R18_or_mu150anc1_a1.pt" --targets "$A/targets_or_mu150.npz" --eps 0.9,0.99 --alphas 1.2,1.4,1.6 --tag R21_null \
    | grep -E "^eps|Error"
  for e in 0.9 0.99; do for a in 1.2 1.4 1.6; do fin "$O/R21_null/eps${e}_alpha_$a.pt" R21_null_e${e}_a$a.pt; done; done ) > "$L/r10_g1.log" 2>&1 &
# G2: repair after extrapolation: 1 short epoch from the pushed model with strong anchors (KL at T=2) + forget targets
( export CUDA_VISIBLE_DEVICES=2
  python "$C/unlearn.py" --targets "$A/targets_or_mu150.npz" --init "$SUB/R19_x150anc1_a1.2.pt" --epochs 1 --lr 5e-6 --w_anchor 3.0 \
    --alphas 1.0,1.15 --tag R21_rep | grep -E "^alpha|Error"
  for a in 1 1.15; do fin "$O/R21_rep/alpha_$a.pt" R21_rep_a$a.pt; done ) > "$L/r10_g2.log" 2>&1 &
# G3: ensemble-like softer non-true shape (original ranking, tail temperature 1.5) + anchors, then push
( export CUDA_VISIBLE_DEVICES=3
  python "$C/targets.py" --class_mode sample --mu 15.0 --b 1.5 --rest orig --tau 1.5 --floor 1e-6 --tag or_mu150t15 | grep -E "target_ce|target_acc"
  python "$C/unlearn.py" --targets "$A/targets_or_mu150t15.npz" --epochs 3 --w_anchor 1.0 --alphas 1.0 --tag R21_t15 | grep -E "^alpha|Error"
  python "$C/extrap.py" --src "$O/R21_t15/alpha_1.pt" --targets "$A/targets_or_mu150t15.npz" --alphas 1.2,1.35 --tag R21_xt15 | grep -E "^alpha|Error"
  for a in 1.2 1.35; do fin "$O/R21_xt15/alpha_$a.pt" R21_xt15_a$a.pt; done ) > "$L/r10_g3.log" 2>&1 &
wait
echo "files: $(find "$W" -xdev | wc -l)"; df --output=iavail /p/scratch/training2666 | tail -1; echo ALLDONE
