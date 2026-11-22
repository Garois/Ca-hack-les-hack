#!/bin/bash
# Inside the team job: GPU=<k> srun --jobid=<id> --overlap -n1 -c60 slurm/pipeline_s2.sh <method>
# synth2 generation (fixed transplant model) -> message features -> U-Net (aux) on fold 0
set -e
cd "$(dirname "$0")/.."
m=$1
./run.sh synth.py --method $m --seed 21 --n 3500 --dir synth2 > logs/s2-synth-$m-21.out 2>&1 &
./run.sh synth.py --method $m --seed 22 --n 3500 --dir synth2 > logs/s2-synth-$m-22.out 2>&1 &
wait
./run.sh feats.py --method $m --synth_dir synth2 --name feats2 > logs/s2-feats-$m.out 2>&1
./run.sh train_seg.py --method $m --fold 0 --epochs 10 --aux --synth_dir synth2 --feats feats2 --workers 12 --tag _s2 > logs/s2-train-$m-f0.out 2>&1
echo "pipeline done $m"
