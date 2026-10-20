#!/usr/bin/env bash
set -euo pipefail
base=/root/autodl-tmp/F2_1_r1_20261019
exec > "$base/A0_retry1.log" 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$base/A0_retry1.exit"' EXIT
export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH="$base/source/src"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "$base/source"
python -u scripts/retry_f2_1_a0.py
