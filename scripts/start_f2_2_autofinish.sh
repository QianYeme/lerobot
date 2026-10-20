#!/usr/bin/env bash
set -euo pipefail
task_root=/root/autodl-tmp/F2_2_r0_20261019
cd "$task_root/source"
export PYTHONPATH=src
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
/root/miniconda3/bin/python -u scripts/autofinish_f2_2.py --base "$task_root" > "$task_root/autofinish.log" 2>&1
