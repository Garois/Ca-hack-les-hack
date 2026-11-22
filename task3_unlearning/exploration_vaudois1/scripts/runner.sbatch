#!/bin/bash
# Runner vaudois1 sur dc-gpu-devel (HORS reservation cispahack : ne prend pas de noeud aux autres equipes).
# Execute les fichiers queue/*.sh (une commande chacun) sur les 4 GPU. S'arrete tout seul :
#   - des qu'un job de l'equipe tourne dans la reservation (=> jamais 2 noeuds pour l'equipe),
#   - si queue/STOP existe, ou apres IDLE_MAX secondes sans rien a faire.
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=01:05:00
#SBATCH --job-name=vau_runner
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
cd "$SLURM_SUBMIT_DIR" || exit 1
source ../.venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
TEAMDIR="/p/scratch/training2666/denoncin1/Ca hack les hack"
IDLE_MAX=${IDLE_MAX:-600}
mkdir -p queue/done logs
declare -A PIDS
idle=0
echo "$(date +%T) runner start on $(hostname)"
while true; do
  if squeue -r --reservation=cispahack -h -t R -o "%Z" 2>/dev/null | grep -qF "$TEAMDIR"; then
    echo "$(date +%T) team reservation job RUNNING -> stop"; kill "${PIDS[@]}" 2>/dev/null; break; fi
  [ -f queue/STOP ] && { echo "$(date +%T) STOP file"; break; }
  busy=0
  for g in 0 1 2 3; do
    pid=${PIDS[$g]}
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then busy=1; continue; fi
    next=$(ls queue/*.sh 2>/dev/null | sort | head -1)
    if [ -n "$next" ]; then
      name=$(basename "$next" .sh); mv "$next" "queue/done/$name.sh"
      CUDA_VISIBLE_DEVICES=$g bash "queue/done/$name.sh" > "logs/$name.log" 2>&1 &
      PIDS[$g]=$!; busy=1
      echo "$(date +%T) start $name on GPU $g (pid ${PIDS[$g]})"
    fi
  done
  if [ $busy -eq 0 ]; then idle=$((idle+10)); else idle=0; fi
  [ $idle -ge $IDLE_MAX ] && { echo "$(date +%T) idle ${IDLE_MAX}s -> exit"; break; }
  sleep 10
done
wait
echo "$(date +%T) runner end"
