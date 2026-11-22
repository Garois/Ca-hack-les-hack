#!/bin/bash
#SBATCH --job-name=unl_round12
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --exclude=jrc[0198,0215,0217,0231-0233,0245,0257-0258,0283,0314,0321,0323-0324,0327,0329,0347,0349-0350,0362-0363,0374,0387,0394,0396,0410,0414,0423,0442,0444]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=logs/round12-%j.out
#SBATCH --error=logs/round12-%j.out
SCR="/p/scratch/training2666/denoncin1/Ca hack les hack/unlearning"
W="$SCR/lemasnedechermont1"
export TASK_DIR="$SCR" UNL_WORK="$W" UNL_ANCHORS="$W/anchors/anchors.npz" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
source "$SCR/.venv/bin/activate"
C="$W/code"; O="$W/output"; A="$W/artifacts"; SUB="$W/submit"; L="$W/logs"; T150="$A/targets_or_mu150.npz"
fin() { python "$C/validate_submission.py" "$1" && cp "$1" "$SUB/$2" && echo "READY $2 $(sha256sum "$SUB/$2" | cut -c1-16)"; }
# G0: null-space projection of the strength-2 anchored model, pushed to the peak area
( export CUDA_VISIBLE_DEVICES=0
  python "$C/nullproj.py" --src "$O/R22_mu150anc2/alpha_1.pt" --targets "$T150" --eps 0.9 --alphas 1.6,1.8,2.0,2.2 --tag R23_null2 | grep -E "keeps|^eps|Error"
  for a in 1.6 1.8 2 2.2; do fin "$O/R23_null2/eps0.9_alpha_$a.pt" R23_null2_e0.9_a$a.pt; done ) > "$L/r12_g0.log" 2>&1 &
# G1: finer push of the strength-2 anchored model around the peak
( export CUDA_VISIBLE_DEVICES=1
  python "$C/extrap.py" --src "$O/R22_mu150anc2/alpha_1.pt" --targets "$T150" --alphas 1.3,1.35,1.45,1.5 --tag R23_x2 | grep -E "^alpha|Error"
  for a in 1.3 1.35 1.45 1.5; do fin "$O/R23_x2/alpha_$a.pt" R23_x150anc2_a$a.pt; done ) > "$L/r12_g1.log" 2>&1 &
# G2: 70%-gate check (80/20 fold-out) for the strength-2 anchored family, plain push and projection
( export CUDA_VISIBLE_DEVICES=2
  python "$C/unlearn.py" --targets "$T150" --epochs 3 --w_anchor 2.0 --folds 80 --alphas 1.0 --tag R23_fold2 | grep -E "^alpha|Error"
  python "$C/extrap.py" --src "$O/R23_fold2/alpha_1.pt" --targets "$T150" --alphas 1.4,1.5 --folds 80 --save 0 --tag R23_xfold2 | grep -E "^alpha|Error"
  python "$C/nullproj.py" --src "$O/R23_fold2/alpha_1.pt" --targets "$T150" --eps 0.9 --alphas 1.8,2.0 --folds 80 --save 0 --tag R23_nfold2 | grep -E "^eps|Error" ) > "$L/r12_g2.log" 2>&1 &
# G3: even stronger anchors (weight 3), pushed
( export CUDA_VISIBLE_DEVICES=3
  python "$C/unlearn.py" --targets "$T150" --epochs 3 --w_anchor 3.0 --alphas 1.0 --tag R23_mu150anc3 | grep -E "^alpha|Error"
  python "$C/extrap.py" --src "$O/R23_mu150anc3/alpha_1.pt" --targets "$T150" --alphas 1.4,1.6,1.8 --tag R23_x3 | grep -E "^alpha|Error"
  for a in 1.4 1.6 1.8; do fin "$O/R23_x3/alpha_$a.pt" R23_x150anc3_a$a.pt; done ) > "$L/r12_g3.log" 2>&1 &
wait
echo "files: $(find "$W" -xdev | wc -l)"; df --output=iavail /p/scratch/training2666 | tail -1; echo ALLDONE
