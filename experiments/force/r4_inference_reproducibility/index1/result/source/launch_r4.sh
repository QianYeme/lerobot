#!/usr/bin/env bash
set -euo pipefail
work=/root/autodl-tmp/R4_20261023
parent=/root/autodl-tmp/F5_1_r3_20261022
export PYTHONPATH="$parent:$parent/recipe:$parent/source/src"
export TORCH_HOME=/root/autodl-tmp/F3_r1_transfer_20261020/bundle/torch_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec /root/miniconda3/bin/python -u "$work/run_r4_repro.py" "$@"
