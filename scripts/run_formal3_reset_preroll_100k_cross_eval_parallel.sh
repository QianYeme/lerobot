#!/usr/bin/env bash
set -euo pipefail

# B1 post-train: time-trim cross eval of the five 100k checkpoints (task
# interval MAE evidence; trim-start observation = the reference frame 0 for
# the original data line).

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/bin:$PATH"

EPISODES="9,11,17,25,26,28,30,31,43,52,55,59"
DATA="$PROJECT/formal3_data/kind_merged_nomaster_time_trim_fit48"
OUT="$PROJECT/outputs/formal3_preroll_p4_100k_20260929/eval_cross_on_timetrim"
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 1; }
mkdir -p "$OUT/logs"

declare -A pids
for step in 020000 040000 060000 080000 100000; do
  label="RESET_PREROLL_ON_TIMETRIM_${step}"
  python -u src/lerobot/scripts/offline_eval_act_sequence.py \
    --checkpoint "$PROJECT/outputs/train/F3_P4_DET_RESET_PREROLL_100K_s1000/checkpoints/$step/pretrained_model" \
    --dataset-root "$DATA" --repo-id QYyyyyyyy/formal3_kind_merged_nomaster_time_trim_fit48 \
    --episodes "$EPISODES" --max-frames 2 --output "$OUT/$label" --device cuda \
    > "$OUT/logs/$label.log" 2>&1 &
  pids[$step]=$!
done

failed=0
for step in "${!pids[@]}"; do
  if wait "${pids[$step]}"; then code=0; else code=$?; failed=1; fi
  echo "$code" > "$OUT/logs/RESET_PREROLL_ON_TIMETRIM_${step}.exit"
done
[[ "$failed" -eq 0 ]] || exit 1
printf 'exit=0\n' > "$OUT/evaluation.done"
