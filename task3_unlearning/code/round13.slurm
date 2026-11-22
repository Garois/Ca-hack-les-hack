#!/bin/bash
#SBATCH --job-name=unl_round13
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --exclude=jrc[0198,0215,0217,0231-0233,0245,0257-0258,0283,0314,0321,0323-0324,0327,0329,0347,0349-0350,0362-0363,0374,0387,0394,0396,0410,0414,0423,0442,0444]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=logs/round13-%j.out
#SBATCH --error=logs/round13-%j.out
SCR="/p/scratch/training2666/denoncin1/Ca hack les hack/unlearning"
W="$SCR/lemasnedechermont1"
export TASK_DIR="$SCR" UNL_WORK="$W" UNL_ANCHORS="$W/anchors/anchors.npz" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
source "$SCR/.venv/bin/activate"
C="$W/code"; O="$W/output"; A="$W/artifacts"; SUB="$W/submit"; L="$W/logs"; T150="$A/targets_or_mu150.npz"
fin() { python "$C/validate_submission.py" "$1" && cp "$1" "$SUB/$2" && echo "READY $2 $(sha256sum "$SUB/$2" | cut -c1-16)"; }
( export CUDA_VISIBLE_DEVICES=0   # anchors weight 5, pushed
  python "$C/unlearn.py" --targets "$T150" --epochs 3 --w_anchor 5.0 --alphas 1.0 --tag R24_mu150anc5 | grep -E "^alpha|Error"
  python "$C/extrap.py" --src "$O/R24_mu150anc5/alpha_1.pt" --targets "$T150" --alphas 1.5,1.7,1.9 --tag R24_x5 | grep -E "^alpha|Error"
  for a in 1.5 1.7 1.9; do fin "$O/R24_x5/alpha_$a.pt" R24_x150anc5_a$a.pt; done ) > "$L/r13_g0.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=1   # null-space projection of the anchors-3 model
  python "$C/nullproj.py" --src "$O/R23_mu150anc3/alpha_1.pt" --targets "$T150" --eps 0.9 --alphas 1.8,2.0,2.2 --tag R24_null3 | grep -E "keeps|^eps|Error"
  for a in 1.8 2 2.2; do fin "$O/R24_null3/eps0.9_alpha_$a.pt" R24_null3_e0.9_a$a.pt; done ) > "$L/r13_g1.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=2   # strength in the targets (mu 20) with anchors 3, mild push
  python "$C/targets.py" --class_mode sample --mu 20.0 --b 1.5 --rest orig --floor 1e-7 --tag or_mu200 | grep -E "target_ce|target_acc"
  python "$C/unlearn.py" --targets "$A/targets_or_mu200.npz" --epochs 3 --w_anchor 3.0 --alphas 1.0 --tag R24_mu200anc3 | grep -E "^alpha|Error"
  python "$C/extrap.py" --src "$O/R24_mu200anc3/alpha_1.pt" --targets "$A/targets_or_mu200.npz" --alphas 1.3,1.5 --tag R24_x200 | grep -E "^alpha|Error"
  for a in 1.3 1.5; do fin "$O/R24_x200/alpha_$a.pt" R24_x200anc3_a$a.pt; done ) > "$L/r13_g2.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=3   # projection strength variants on the anchors-2 model
  python "$C/nullproj.py" --src "$O/R22_mu150anc2/alpha_1.pt" --targets "$T150" --eps 0.8,0.95 --alphas 1.7,1.9 --tag R24_null2b | grep -E "keeps|^eps|Error"
  for e in 0.8 0.95; do for a in 1.7 1.9; do fin "$O/R24_null2b/eps${e}_alpha_$a.pt" R24_null2_e${e}_a$a.pt; done; done ) > "$L/r13_g3.log" 2>&1 &
wait
echo "files: $(find "$W" -xdev | wc -l)"; df --output=iavail /p/scratch/training2666 | tail -1; echo ALLDONE
