#!/usr/bin/env bash
set -euo pipefail
base=/root/autodl-tmp/F2_1_r1_20261019
exec > "$base/gates.log" 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$base/gates.exit"' EXIT
export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH="$base/source/src"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "$base/source"
python scripts/run_f2_1.py --root "$base/inputs" --data /root/autodl-tmp/lerobot/formal4_data/kind_merged_nomaster --out "$base/results" --mode smoke
if [[ ! -f "$base/results/historical_metric_reproduction.json" ]]; then
    python scripts/run_f2_1.py --root "$base/inputs" --data /root/autodl-tmp/lerobot/formal4_data/kind_merged_nomaster --out "$base/results" --mode historical --historical-checkpoint /root/autodl-tmp/lerobot/results/experiments/T0_POSITION_DISCOVERY_ACT_VS_OCACT/P1_ocfix2_s1/checkpoints/100000/pretrained_model
fi
echo 'GATES_COMPLETE. Training is launched separately only after resource review.'
