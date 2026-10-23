#!/usr/bin/env bash
set -euo pipefail
work=/root/autodl-tmp/R5_20261023
parent=/root/autodl-tmp/F5_1_r3_20261022
export PYTHONPATH="$work/source/src:$parent:$parent/recipe"
export TORCH_HOME=/root/autodl-tmp/F5_20261022/bundle/torch_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
/root/miniconda3/bin/python -u "$work/source/scripts/build_r5_bundle.py"
/root/miniconda3/bin/python -u "$work/source/scripts/smoke_r5_runtime.py"
printf '0\n' > "$work/prepare.exit"
