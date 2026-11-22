#!/bin/bash -l
# usage: [GPU=k] ours/run.sh script.py [args]  -> runs inside ours/ with the team venv
# The venv was built on the JURECA Python module: without it, python picks the system libpython and segfaults.
# Team rule (organisers' cluster rules): ONE job per team in the cispahack reservation. The job owns the whole node (4 GPUs),
# so run several things inside it with srun --jobid=<id> --overlap -n1 -cN and pick a GPU with GPU=0..3.
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3 >/dev/null 2>&1
cd "$(dirname "$0")"
[ -n "$GPU" ] && export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1 HF_HOME="$(pwd)/../.cache/huggingface" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
exec ../venv/bin/python "$@"
