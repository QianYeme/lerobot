#!/usr/bin/env bash
# Phase-C screening: chronological action-chunk capture for one model's short-run checkpoints on dev8.
# Usage: run_phaseC_dev8_screening.sh det|i0|i1 [steps...]   (default steps: 002000 005000 010000)
set -u
cd /root/autodl-tmp/lerobot/lerobot-main
OUT=outputs/act_sequence_phaseC_2026-09-18
DEV8=5,6,8,24,30,33,36,39
MODEL="${1:?Usage: run_phaseC_dev8_screening.sh det|i0|i1 [steps...]}"
shift
if [[ $# -gt 0 ]]; then STEPS=("$@"); else STEPS=(002000 005000 010000); fi
case "$MODEL" in det) NAME=DET ;; i0) NAME=I0 ;; i1) NAME=I1 ;; *) exit 2 ;; esac
mkdir -p "$OUT"
exec > "$OUT/run_${MODEL}.log" 2>&1
echo "START ${NAME} $(date '+%F %T') host=$(hostname)"
for S in "${STEPS[@]}"; do
  TARGET="$OUT/C50_${NAME}_dev8_s${S}"
  if [[ -e "$TARGET" ]]; then echo "exists $TARGET; skip"; continue; fi
  CKPT="outputs/train/C50_NOMASTER_${NAME}_short_s1000/checkpoints/${S}/pretrained_model"
  echo "=== start ${NAME} ${S} $(date +%H:%M:%S) ==="
  /root/miniconda3/bin/python src/lerobot/scripts/offline_eval_act_sequence.py \
    --checkpoint "$CKPT" \
    --dataset-root 数据集/formal1_C50_nomaster \
    --repo-id QYyyyyyyy/formal1_C50_nomaster \
    --episodes "$DEV8" \
    --output "$TARGET"
  echo "=== end ${NAME} ${S} rc=$? $(date +%H:%M:%S) ==="
done
echo "ALL_DONE ${NAME} $(date '+%F %T')"
