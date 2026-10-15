#!/usr/bin/env bash
set -eu
ulimit -n 65535
cd /root/autodl-tmp/lerobot
export PYTHONPATH=/root/autodl-tmp/lerobot/src
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
/root/miniconda3/bin/python scripts/eval_d2_action_chain.py \
 --checkpoint results/experiments/N0_OVERNIGHT_SPATIAL_CONDITIONING_BATCH/R3_FILM_s1/checkpoints/100000/pretrained_model \
 --root formal4_data/kind_merged_nomaster \
 --split experiments/data/d1_training_readiness/index1/result/split_manifest_v1.json \
 --out results/experiments/D2_P1_ACTION_CHAIN_DIAGNOSIS/R3 \
 > results/experiments/D2_P1_ACTION_CHAIN_DIAGNOSIS/r3.log 2>&1
touch results/experiments/D2_P1_ACTION_CHAIN_DIAGNOSIS/r3.done
