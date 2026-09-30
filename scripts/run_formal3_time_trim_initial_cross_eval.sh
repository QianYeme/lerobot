#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/bin:$PATH"

EPISODES="9,11,17,25,26,28,30,31,43,52,55,59"
DATA="$PROJECT/数据集/formal3/kind_merged_nomaster_fit48"
OUT="$PROJECT/outputs/formal3_time_trim_p3_20260928/eval_initial_cross"
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 1; }
mkdir -p "$OUT/logs"

for step in 002000 004000 006000 008000 010000; do
  label="TIMETRIM_ON_ORIGINAL_${step}"
  if python -u src/lerobot/scripts/offline_eval_act_sequence.py \
    --checkpoint "$PROJECT/outputs/train/F3_P3_DET_TIMETRIM_10K_s1000/checkpoints/$step/pretrained_model" \
    --dataset-root "$DATA" --repo-id QYyyyyyyy/formal3_kind_merged_nomaster_fit48 \
    --episodes "$EPISODES" --max-frames 2 --output "$OUT/$label" --device cuda \
    > "$OUT/logs/$label.log" 2>&1; then
    echo 0 > "$OUT/logs/$label.exit"
  else
    echo $? > "$OUT/logs/$label.exit"
    exit 1
  fi
done
printf 'exit=0\n' > "$OUT/evaluation.done"
