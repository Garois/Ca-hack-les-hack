#!/bin/bash
# submit gpujob.sh "$1" only when the team has no job at all (running or queued); one-shot.
cd "/p/scratch/training2666/denoncin1/Ca hack les hack/model_equality/work"
T=$(pwd | grep -oE '/p/scratch/training2666/[^/]+/[^/]+')
while true; do
  n=$(squeue -h -o "%Z" | grep -c "$T")
  if [ "$n" = 0 ]; then
    sleep 20
    n=$(squeue -h -o "%Z" | grep -c "$T")
    if [ "$n" = 0 ]; then sbatch gpujob.sh "$1"; date; exit 0; fi
  fi
  sleep 60
done
