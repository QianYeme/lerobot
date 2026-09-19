#!/usr/bin/env bash
# Phase-E evaluation: chronological action-chunk capture for one 100k full-run checkpoint on fixed val10.
# Usage: run_phaseE_val10_screening.sh act|det|i0|i1 [steps...]   (default steps: 100000)
set -u
cd /root/autodl-tmp/lerobot/lerobot-main
OUT=outputs/act_sequence_phaseE_2026-09-19
VAL10=7,11,13,14,16,27,35,38,40,43
MODEL="${1:?Usage: run_phaseE_val10_screening.sh act|det|i0|i1 [steps...]}"
shift
if [[ $# -gt 0 ]]; then STEPS=("$@"); else STEPS=(100000); fi
case "$MODEL" in act) NAME=ACT ;; det) NAME=DET ;; i0) NAME=I0 ;; i1) NAME=I1 ;; *) exit 2 ;; esac
mkdir -p "$OUT"
exec > "$OUT/run_${MODEL}.log" 2>&1
echo "START ${NAME} $(date '+%F %T') host=$(hostname)"
for S in "${STEPS[@]}"; do
  TARGET="$OUT/C50_${NAME}_val10_s${S}"
  if [[ -e "$TARGET" ]]; then echo "exists $TARGET; skip"; continue; fi
  CKPT="outputs/train/C50_NOMASTER_${NAME}_s1000/checkpoints/${S}/pretrained_model"
  echo "=== start ${NAME} ${S} $(date +%H:%M:%S) ==="
  /root/miniconda3/bin/python src/lerobot/scripts/offline_eval_act_sequence.py \
    --checkpoint "$CKPT" \
    --dataset-root 数据集/formal1_C50_nomaster \
    --repo-id QYyyyyyyy/formal1_C50_nomaster \
    --episodes "$VAL10" \
    --output "$TARGET"
  echo "=== end ${NAME} ${S} rc=$? $(date +%H:%M:%S) ==="
done
echo "ALL_DONE ${NAME} $(date '+%F %T')"
