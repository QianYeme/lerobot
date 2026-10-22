#!/usr/bin/env bash
set -euo pipefail
work=/root/autodl-tmp/F5_20261022
export PYTHONPATH="$work/source/src":/root/autodl-tmp/F4_1_20261022:/root/autodl-tmp/F2_1_r1_20261019/source/src
export TORCH_HOME=/root/autodl-tmp/F3_r1_transfer_20261020/bundle/torch_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ "${1:-launch}" == launch ]]; then
    test ! -e "$work/prepare.log"
    nohup bash "$work/prepare.sh" run > "$work/prepare.log" 2>&1 < /dev/null &
    printf '%s\n' "$!"
    exit 0
fi
trap 'printf "%s\n" "$?" > "$work/prepare.exit"' EXIT
/root/miniconda3/bin/python -u "$work/build_f5_bundle.py"
/root/miniconda3/bin/python -u "$work/smoke_f5_runtime.py"
