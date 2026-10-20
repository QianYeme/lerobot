#!/usr/bin/env bash
set -euo pipefail
task_root=/root/autodl-tmp/F2_2_r0_20261019
cd "$task_root/source"
export PYTHONPATH=src
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
for variant in B0 B1; do
  test ! -e "$task_root/results/$variant"
  screen -dmS "F22_$variant" bash "$task_root/source/scripts/train_f2_2_one.sh" "$variant"
done
screen -ls
