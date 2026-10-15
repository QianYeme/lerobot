#!/usr/bin/env bash
set -eu
ulimit -n 65535
cd /root/autodl-tmp/lerobot
export PYTHONPATH=/root/autodl-tmp/lerobot/src
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
mkdir -p results/experiments/D2_P1_ACTION_CHAIN_DIAGNOSIS
if [ -f results/experiments/D2_P1_ACTION_CHAIN_DIAGNOSIS/p1.log ]; then
  cp results/experiments/D2_P1_ACTION_CHAIN_DIAGNOSIS/p1.log results/experiments/D2_P1_ACTION_CHAIN_DIAGNOSIS/p1_failed_open_files.log
fi
git rev-parse HEAD > results/experiments/D2_P1_ACTION_CHAIN_DIAGNOSIS/remote_commit.txt
git diff > results/experiments/D2_P1_ACTION_CHAIN_DIAGNOSIS/remote_dirty.patch
/root/miniconda3/bin/python scripts/eval_d2_action_chain.py \
  --checkpoint results/experiments/T0_POSITION_DISCOVERY_ACT_VS_OCACT/P1_ocfix2_s1/checkpoints/100000/pretrained_model \
  --root formal4_data/kind_merged_nomaster \
  --split experiments/data/d1_training_readiness/index1/result/split_manifest_v1.json \
  --out results/experiments/D2_P1_ACTION_CHAIN_DIAGNOSIS/P1 \
  > results/experiments/D2_P1_ACTION_CHAIN_DIAGNOSIS/p1.log 2>&1
touch results/experiments/D2_P1_ACTION_CHAIN_DIAGNOSIS/p1.done
