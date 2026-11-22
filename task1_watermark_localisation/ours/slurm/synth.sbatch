#!/bin/bash
#SBATCH --account=training2666
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# usage (from ours/): sbatch -J synth-<method>-<seed> slurm/synth.sbatch <method> <seed> <n>
./run.sh synth.py --method "$1" --seed "$2" --n "$3"
