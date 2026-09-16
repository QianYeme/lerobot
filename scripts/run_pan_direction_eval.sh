#!/usr/bin/env bash
# Pan-direction-consistency evaluation for the three 100k C50 models.
# Run inside a screen on the training server:
#   screen -S pan_direction -dm bash scripts/run_pan_direction_eval.sh
set -uo pipefail

PROJECT=/root/autodl-tmp/lerobot/lerobot-main
cd "$PROJECT" || exit 1

PY=/root/miniconda3/bin/python
OUT=outputs/formal1_C50_phase1/pan_direction_100k
mkdir -p "$OUT"
: > "$OUT/exit_codes.txt"

EPISODES=7,11,13,14,16,27,35,38,40,43
DATA_ROOT=数据集/formal1_C
ANN=数据集/formal1_C/annotations

run_model () {
  local name="$1"
  "$PY" src/lerobot/scripts/eval_pan_direction_consistency.py \
    --checkpoint "outputs/train/${name}/checkpoints/100000/pretrained_model" \
    --dataset.repo_id QYyyyyyyy/formal1_C \
    --dataset.root "$DATA_ROOT" \
    --annotation-dir "$ANN" \
    --episodes "$EPISODES" \
    --batch-size 8 --max-batches 100 --action-steps 10 --seed 1000 \
    --output "$OUT/${name}_pan_direction.json" \
    > "$OUT/${name}_pan_direction.log" 2>&1
  echo "$name exit=$?" >> "$OUT/exit_codes.txt"
}

# One wave of three models in parallel (same pattern as the visual-sensitivity
# runs: three AV1 decoders run without measurable contention).
pids=()
for name in C50_ACT_s1000 C50_DET_s1000 C50_INJECT_s1000; do
  run_model "$name" &
  pids+=($!)
done
wait

echo done > "$OUT/COMPLETE"
