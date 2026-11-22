#!/bin/bash
#SBATCH --job-name=unl_round20
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --exclude=jrc[0198,0215,0217,0231-0233,0245,0257-0258,0283,0314,0321,0323-0324,0327,0329,0347,0349-0350,0362-0363,0374,0387,0394,0396,0410,0414,0423,0442,0444]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=logs/round20-%j.out
#SBATCH --error=logs/round20-%j.out
SCR="/p/scratch/training2666/denoncin1/Ca hack les hack/unlearning"
W="$SCR/lemasnedechermont1"
export TASK_DIR="$SCR" UNL_WORK="$W" UNL_ANCHORS="$W/anchors/anchors.npz" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
source "$SCR/.venv/bin/activate"
C="$W/code"; O="$W/output"; A="$W/artifacts"; SUB="$W/submit"; L="$W/logs"
fin() { python "$C/validate_submission.py" "$1" && cp "$1" "$SUB/$2" && echo "READY $2 $(sha256sum "$SUB/$2" | cut -c1-16)"; }
run() { # tag, targets-tag, alphas
  python "$C/unlearn.py" --targets "$A/targets_$2.npz" --epochs 3 --w_anchor 5.0 --alphas 1.0 --tag R31_$1 | grep -E "^alpha|Error" | cut -c1-200
  python "$C/nullproj.py" --src "$O/R31_$1/alpha_1.pt" --targets "$A/targets_$2.npz" --eps 0.9 --alphas $3 --tag R31_n$1 | grep -E "keeps|^eps|Error"
  for a in ${3//,/ }; do a2=$(python -c "print('%g' % $a)"); fin "$O/R31_n$1/eps0.9_alpha_$a2.pt" R31_n$1_e0.9_a$a2.pt; done; }
( export CUDA_VISIBLE_DEVICES=0; run t15 or_mu150t15 2.5,2.65,2.8 ) > "$L/r20_g0.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=1
  python "$C/targets.py" --class_mode sample --mu 15.0 --b 1.5 --rest orig --tau 1.25 --floor 1e-6 --tag or_mu150t125 | grep -E "target_ce|target_acc"
  run t125 or_mu150t125 2.5,2.65,2.8 ) > "$L/r20_g1.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=2
  python "$C/targets.py" --class_mode sample --mu 15.0 --b 0 --rest orig --floor 1e-7 --tag or_mu150b0 | grep -E "target_ce|target_acc"
  run b0 or_mu150b0 2.5,2.65,2.8 ) > "$L/r20_g2.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=3
  python "$C/targets.py" --class_mode sample --mu 15.0 --b 3.0 --rest orig --floor 1e-7 --tag or_mu150b3 | grep -E "target_ce|target_acc"
  run b3 or_mu150b3 2.5,2.65,2.8 ) > "$L/r20_g3.log" 2>&1 &
wait; echo "files: $(find "$W" -xdev | wc -l)"; df --output=iavail /p/scratch/training2666 | tail -1; echo ALLDONE
