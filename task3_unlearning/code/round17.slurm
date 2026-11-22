#!/bin/bash
#SBATCH --job-name=unl_round17
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --exclude=jrc[0198,0215,0217,0231-0233,0245,0257-0258,0283,0314,0321,0323-0324,0327,0329,0347,0349-0350,0362-0363,0374,0387,0394,0396,0410,0414,0423,0442,0444]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=logs/round17-%j.out
#SBATCH --error=logs/round17-%j.out
SCR="/p/scratch/training2666/denoncin1/Ca hack les hack/unlearning"
W="$SCR/lemasnedechermont1"
export TASK_DIR="$SCR" UNL_WORK="$W" UNL_ANCHORS="$W/anchors/anchors.npz" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
source "$SCR/.venv/bin/activate"
C="$W/code"; O="$W/output"; A="$W/artifacts"; SUB="$W/submit"; L="$W/logs"; T150="$A/targets_or_mu150.npz"
fin() { python "$C/validate_submission.py" "$1" && cp "$1" "$SUB/$2" && echo "READY $2 $(sha256sum "$SUB/$2" | cut -c1-16)"; }
np_() { python "$C/nullproj.py" --src "$O/$1/alpha_1.pt" --targets "$T150" --eps $2 --alphas $3 --tag $4 | grep -E "keeps|^eps|Error"
        for e in ${2//,/ }; do for a in ${3//,/ }; do a2=$(python -c "print('%g' % $a)"); fin "$O/$4/eps${e}_alpha_$a2.pt" $4_e${e}_a$a2.pt; done; done; }
soupf() { python "$C/soup.py" --srcs "$2" --targets "$T150" --tag $1 | grep -E "^soup|Error"; fin "$O/$1/alpha_1.pt" $1.pt; }
( export CUDA_VISIBLE_DEVICES=0; np_ R24_mu150anc5 0.97 2.8,3.0,3.2 R28_null5i ) > "$L/r17_g0.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; np_ R24_mu150anc5 0.99 3.0,3.3,3.6 R28_null5j ) > "$L/r17_g1.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=2; np_ R24_mu150anc5 0.85 2.5,2.65 R28_null5k ) > "$L/r17_g2.log" 2>&1 &
( export CUDA_VISIBLE_DEVICES=3
  python "$C/unlearn.py" --targets "$T150" --epochs 3 --w_anchor 5.0 --seed 2 --alphas 1.0 --tag R28_mu150anc5s2 | grep -E "^alpha|Error"
  np_ R28_mu150anc5s2 0.9 2.65,2.7 R28_null5s2
  soupf R28_soup3_a2.65 "$SUB/R26_null5f_e0.9_a2.65.pt,$SUB/R27_null5s1_e0.9_a2.65.pt,$SUB/R28_null5s2_e0.9_a2.65.pt" ) > "$L/r17_g3.log" 2>&1 &
wait; echo "files: $(find "$W" -xdev | wc -l)"; df --output=iavail /p/scratch/training2666 | tail -1; echo ALLDONE
