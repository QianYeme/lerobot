#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
[[ "$MODE" == smoke || "$MODE" == full ]] || {
  echo 'usage: run_formal3_mixed_dev12_eval.sh [smoke|full]' >&2
  exit 2
}

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/bin:$PATH"

DATA_ROOT="$PROJECT/数据集/formal3/kind_merged_nomaster_fit48"
REPO_ID="QYyyyyyyy/formal3_kind_merged_nomaster_fit48"
EPISODES="9,11,17,25,26,28,30,31,43,52,55,59"
CAMPAIGN="$PROJECT/outputs/formal3_mixed_p4_20260927"
if [[ "$MODE" == smoke ]]; then
  OUT="$CAMPAIGN/eval_dev12_smoke"
  EXTRA=(--max-frames 2)
else
  OUT="$CAMPAIGN/eval_dev12"
  EXTRA=()
fi
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 1; }
mkdir -p "$OUT/logs"

run_family() {
  local name="$1" train_dir="$2"
  local failed=0
  for step in 020000 040000 060000 080000 100000; do
    local label="${name}_${step}"
    if python -u src/lerobot/scripts/offline_eval_act_sequence.py \
      --checkpoint "$PROJECT/outputs/train/$train_dir/checkpoints/$step/pretrained_model" \
      --dataset-root "$DATA_ROOT" --repo-id "$REPO_ID" --episodes "$EPISODES" \
      --output "$OUT/$label" --device cuda "${EXTRA[@]}" \
      > "$OUT/logs/$label.log" 2>&1; then
      echo 0 > "$OUT/logs/$label.exit"
    else
      echo $? > "$OUT/logs/$label.exit"
      failed=1
      break
    fi
  done
  return "$failed"
}

run_family DIFFUSION F3_P4_DIFFUSION_BASE_s1000 & p1=$!
run_family DET F3_P4_DET_BASE_s1000 & p2=$!
run_family C1 F3_P4_DET_C1_STATE_s1000 & p3=$!
run_family PHASE F3_P4_DET_PHASE_W010_s1000 & p4=$!

failed=0
for pair in "DIFFUSION:$p1" "DET:$p2" "C1:$p3" "PHASE:$p4"; do
  label="${pair%%:*}"
  pid="${pair##*:}"
  if wait "$pid"; then
    echo 0 > "$OUT/logs/${label}_family.exit"
  else
    code=$?
    echo "$code" > "$OUT/logs/${label}_family.exit"
    failed=1
  fi
done
[[ "$failed" -eq 0 ]] || { echo 'at least one evaluation family failed' >&2; exit 1; }
printf 'exit=0\n' > "$OUT/evaluation.done"
