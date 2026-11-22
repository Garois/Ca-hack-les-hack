#!/bin/bash
#SBATCH --job-name=unl_round18
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --exclude=jrc[0198,0215,0217,0231-0233,0245,0257-0258,0283,0314,0321,0323-0324,0327,0329,0347,0349-0350,0362-0363,0374,0387,0394,0396,0410,0414,0423,0442,0444]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:40:00
#SBATCH --output=logs/round18-%j.out
#SBATCH --error=logs/round18-%j.out
SCR="/p/scratch/training2666/denoncin1/Ca hack les hack/unlearning"
W="$SCR/lemasnedechermont1"
export TASK_DIR="$SCR" UNL_WORK="$W" UNL_ANCHORS="$W/anchors/anchors.npz" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
source "$SCR/.venv/bin/activate"
C="$W/code"; O="$W/output"; A="$W/artifacts"; SUB="$W/submit"; L="$W/logs"; T150="$A/targets_or_mu150.npz"
fin() { python "$C/validate_submission.py" "$1" && cp "$1" "$SUB/$2" && echo "READY $2 $(sha256sum "$SUB/$2" | cut -c1-16)"; }
fold() { python "$C/unlearn.py" --targets "$T150" --epochs 3 --w_anchor 5.0 --folds k5_$1 --w_ho 3.0 --alphas 1.0 --tag R29_f$1 | grep -E "^alpha|Error"; }
# phase 1: 5 fold models (each forgets 80% while the held-out 20% is anchored to orig)
( export CUDA_VISIBLE_DEVICES=0; fold 0; fold 4 ) > "$L/r18_f04.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; fold 1 ) > "$L/r18_f1.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=2; fold 2 ) > "$L/r18_f2.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=3; fold 3 ) > "$L/r18_f3.log" 2>&1 &
wait
# phase 2: soup of the 5 fold models, then projection / push
export CUDA_VISIBLE_DEVICES=0
python "$C/soup.py" --srcs "$O/R29_f0/alpha_1.pt,$O/R29_f1/alpha_1.pt,$O/R29_f2/alpha_1.pt,$O/R29_f3/alpha_1.pt,$O/R29_f4/alpha_1.pt" --targets "$T150" --tag R29_soupF > "$L/r18_soup.log" 2>&1
np_() { python "$C/nullproj.py" --src "$O/R29_soupF/alpha_1.pt" --targets "$T150" --eps $1 --alphas $2 --tag $3 | grep -E "keeps|^eps|Error"
        for a in ${2//,/ }; do a2=$(python -c "print('%g' % $a)"); fin "$O/$3/eps$1_alpha_$a2.pt" $3_e$1_a$a2.pt; done; }
( export CUDA_VISIBLE_DEVICES=0; np_ 0.9 2.6,3.0,3.4 R29_nF ) > "$L/r18_p0.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; np_ 0.9 3.8,4.3 R29_nFb ) > "$L/r18_p1.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=2; python "$C/extrap.py" --src "$O/R29_soupF/alpha_1.pt" --targets "$T150" --alphas 1.5,2.0,2.5 --tag R29_xF | grep -E "^alpha|Error"
  for a in 1.5 2 2.5; do fin "$O/R29_xF/alpha_$a.pt" R29_xF_a$a.pt; done ) > "$L/r18_p2.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=3; np_ 0.95 3.0,3.6 R29_nFc ) > "$L/r18_p3.log" 2>&1 &
wait; echo "files: $(find "$W" -xdev | wc -l)"; df --output=iavail /p/scratch/training2666 | tail -1; echo ALLDONE
