#!/usr/bin/env bash
set -uo pipefail
task_root=/root/autodl-tmp/F2_2_r0_20261019
variant="$1"
cd "$task_root/source"
export PYTHONPATH=src
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
/root/miniconda3/bin/python -u scripts/run_f2_2.py \
  --root "$task_root/inputs" \
  --data /root/autodl-tmp/lerobot/formal4_data/kind_merged_nomaster \
  --out "$task_root/results" --mode train --variant "$variant" > "$task_root/$variant.log" 2>&1
code=$?
printf '%s\n' "$code" > "$task_root/$variant.exit"
exit "$code"
