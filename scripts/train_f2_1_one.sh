#!/usr/bin/env bash
set -euo pipefail
base=/root/autodl-tmp/F2_1_r1_20261019
variant=$1
exec > "$base/$variant.log" 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$base/$variant.exit"' EXIT
export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH="$base/source/src"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "$base/source"
python -u scripts/run_f2_1.py --root "$base/inputs" --data /root/autodl-tmp/lerobot/formal4_data/kind_merged_nomaster --out "$base/results" --mode train --variant "$variant"
