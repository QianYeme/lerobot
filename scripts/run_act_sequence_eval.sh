#!/usr/bin/env bash
# Chronological action-chunk capture + continuity diagnostics for the three C50 100k models.
# Deployment-consistent setting: master_gripper.pos zeroed (raw-master-zero).
set -u
cd /root/autodl-tmp/lerobot/lerobot-main
OUT=outputs/act_sequence_2026-09-18
mkdir -p "$OUT"
exec > "$OUT/run.log" 2>&1
echo "START $(date '+%F %T')  host=$(hostname)"
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
for M in DET ACT INJECT; do
  echo "=== start ${M} $(date +%H:%M:%S) ==="
  /root/miniconda3/bin/python src/lerobot/scripts/offline_eval_act_sequence.py \
    --checkpoint "outputs/train/C50_${M}_s1000/checkpoints/100000/pretrained_model" \
    --dataset-root 数据集/formal1_C \
    --repo-id QYyyyyyyy/formal1_C \
    --raw-master-zero \
    --output "$OUT/C50_${M}_zeromaster"
  echo "=== end ${M} rc=$? $(date +%H:%M:%S) ==="
done
echo "ALL_DONE $(date '+%F %T')"
