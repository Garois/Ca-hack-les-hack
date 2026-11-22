#!/bin/bash
#SBATCH --job-name=unl_round11
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --exclude=jrc[0198,0215,0217,0231-0233,0245,0257-0258,0283,0314,0321,0323-0324,0327,0329,0347,0349-0350,0362-0363,0374,0387,0394,0396,0410,0414,0423,0442,0444]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=logs/round11-%j.out
#SBATCH --error=logs/round11-%j.out
# ONE unreserved node, 4 GPUs: analysis of all scored models + 70%-gate check + stronger anchors
SCR="/p/scratch/training2666/denoncin1/Ca hack les hack/unlearning"
W="$SCR/lemasnedechermont1"
export TASK_DIR="$SCR" UNL_WORK="$W" UNL_ANCHORS="$W/anchors/anchors.npz" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
source "$SCR/.venv/bin/activate"
C="$W/code"; O="$W/output"; A="$W/artifacts"; SUB="$W/submit"; L="$W/logs"
fin() { python "$C/validate_submission.py" "$1" && cp "$1" "$SUB/$2" && echo "READY $2 $(sha256sum "$SUB/$2" | cut -c1-16)"; }
( export CUDA_VISIBLE_DEVICES=0; python "$C/analyze_scored.py" --part 0 --nparts 2 ) > "$L/r11_an0.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; python "$C/analyze_scored.py" --part 1 --nparts 2 ) > "$L/r11_an1.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=2
  python "$C/unlearn.py" --targets "$A/targets_or_mu150.npz" --epochs 3 --w_anchor 1.0 --folds 80 --alphas 1.0 --tag R22_fold150anc1 | grep -E "^alpha|Error"
  python "$C/extrap.py" --src "$O/R22_fold150anc1/alpha_1.pt" --targets "$A/targets_or_mu150.npz" --alphas 1.2,1.35,1.5 --folds 80 --save 0 --tag R22_xfold | grep -E "^alpha|Error" ) > "$L/r11_fold.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=3
  python "$C/unlearn.py" --targets "$A/targets_or_mu150.npz" --epochs 3 --w_anchor 2.0 --alphas 1.0 --tag R22_mu150anc2 | grep -E "^alpha|Error"
  python "$C/extrap.py" --src "$O/R22_mu150anc2/alpha_1.pt" --targets "$A/targets_or_mu150.npz" --alphas 1.2,1.4,1.6 --tag R22_x150anc2 | grep -E "^alpha|Error"
  for a in 1.2 1.4 1.6; do fin "$O/R22_x150anc2/alpha_$a.pt" R22_x150anc2_a$a.pt; done ) > "$L/r11_anc2.log" 2>&1 &
wait
echo "files: $(find "$W" -xdev | wc -l)"; df --output=iavail /p/scratch/training2666 | tail -1; echo ALLDONE
