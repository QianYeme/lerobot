#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
OUT=outputs/mask_inject_phase1_20260917
LOSS_TYPE="${1:-l1}"
case "$LOSS_TYPE" in l1|bce_dice) ;; *) exit 2 ;; esac
NAME=mask_training_check
if [[ "$LOSS_TYPE" != l1 ]]; then NAME="${NAME}_${LOSS_TYPE}"; fi
/root/miniconda3/bin/python -u scripts/check_c50_mask_training.py \
  --dataset-root "$PROJECT/数据集/formal1_C50_nomaster" \
  --mask-dir "$OUT/mask_preview/masks" \
  --reference-checkpoint outputs/train/C50_NOMASTER_DET_smoke_s1000/checkpoints/000020/pretrained_model \
  --output "$OUT/$NAME" --steps 300 --loss-type "$LOSS_TYPE" 2>&1 | tee "$OUT/$NAME.log"
date -Is > "$OUT/$NAME.finished_at.txt"
