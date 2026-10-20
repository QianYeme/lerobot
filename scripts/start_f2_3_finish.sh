#!/usr/bin/env bash
set -euo pipefail
task_root=/root/autodl-tmp/F2_3_r0_20261020
cd "$task_root/source"
export PYTHONPATH=src
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
/root/miniconda3/bin/python -u scripts/finish_f2_3.py > "$task_root/finish.log" 2>&1
