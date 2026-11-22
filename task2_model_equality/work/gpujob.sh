#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=data/slurm-%j.out
# usage: sbatch gpujob.sh "python cmd1 ; python cmd2"
cd "/p/scratch/training2666/denoncin1/Ca hack les hack/model_equality/work"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PATH=$PWD/.venv-llm/bin:$PATH
nvidia-smi -L
eval "$1"
