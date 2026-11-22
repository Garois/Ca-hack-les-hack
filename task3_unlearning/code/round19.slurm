#!/bin/bash
#SBATCH --job-name=unl_round19
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --exclude=jrc[0198,0215,0217,0231-0233,0245,0257-0258,0283,0314,0321,0323-0324,0327,0329,0347,0349-0350,0362-0363,0374,0387,0394,0396,0410,0414,0423,0442,0444]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:40:00
#SBATCH --output=logs/round19-%j.out
#SBATCH --error=logs/round19-%j.out
SCR="/p/scratch/training2666/denoncin1/Ca hack les hack/unlearning"
W="$SCR/lemasnedechermont1"
export TASK_DIR="$SCR" UNL_WORK="$W" UNL_ANCHORS="$W/anchors/anchors.npz" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
source "$SCR/.venv/bin/activate"
C="$W/code"; O="$W/output"; A="$W/artifacts"; SUB="$W/submit"; L="$W/logs"; T150="$A/targets_or_mu150.npz"
fin() { python "$C/validate_submission.py" "$1" && cp "$1" "$SUB/$2" && echo "READY $2 $(sha256sum "$SUB/$2" | cut -c1-16)"; }
np_() { python "$C/nullproj.py" --src "$O/$1/alpha_1.pt" --targets "$T150" --eps $2 --alphas $3 --tag $4 | grep -E "keeps|^eps|Error"
        for a in ${3//,/ }; do a2=$(python -c "print('%g' % $a)"); fin "$O/$4/eps$2_alpha_$a2.pt" $4_e$2_a$a2.pt; done; }
fold() { python "$C/unlearn.py" --targets "$T150" --epochs 3 --w_anchor 5.0 --folds k5_$1 --w_ho 1.0 --alphas 1.0 --tag R30_g$1 | grep -E "^alpha|Error" | cut -c1-200; }
( export CUDA_VISIBLE_DEVICES=0; fold 0; fold 4 ) > "$L/r19_f04.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; fold 1; np_ R29_soupF 0.9 4.8,5.3,5.8 R30_nF ) > "$L/r19_f1.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=2; fold 2 ) > "$L/r19_f2.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=3; fold 3 ) > "$L/r19_f3.log" 2>&1 &
wait
export CUDA_VISIBLE_DEVICES=0
python "$C/soup.py" --srcs "$O/R30_g0/alpha_1.pt,$O/R30_g1/alpha_1.pt,$O/R30_g2/alpha_1.pt,$O/R30_g3/alpha_1.pt,$O/R30_g4/alpha_1.pt" --targets "$T150" --tag R30_soupG > "$L/r19_soup.log" 2>&1
( export CUDA_VISIBLE_DEVICES=0; np_ R30_soupG 0.9 2.6,3.0 R30_nG ) > "$L/r19_p0.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; np_ R30_soupG 0.9 3.4,3.8 R30_nGb ) > "$L/r19_p1.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=2; np_ R30_soupG 0.9 4.2,4.6 R30_nGc ) > "$L/r19_p2.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=3; python "$C/extrap.py" --src "$O/R30_soupG/alpha_1.pt" --targets "$T150" --alphas 2.0,2.5,3.0 --tag R30_xG | grep -E "^alpha|Error"
  for a in 2 2.5 3; do fin "$O/R30_xG/alpha_$a.pt" R30_xG_a$a.pt; done ) > "$L/r19_p3.log" 2>&1 &
wait; echo "files: $(find "$W" -xdev | wc -l)"; df --output=iavail /p/scratch/training2666 | tail -1; echo ALLDONE
