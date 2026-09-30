#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/bin:$PATH"

EPISODES="9,11,17,25,26,28,30,31,43,52,55,59"
CAMPAIGN="$PROJECT/outputs/formal3_reset_pair_p3_20260928"
OUT="$CAMPAIGN/eval_dev12"
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 1; }
mkdir -p "$OUT/logs"

run_family() {
  local family="$1" train_dir="$2" data_root="$3" repo_id="$4"
  for step in 002000 004000 006000 008000 010000; do
    local label="${family}_${step}"
    if python -u src/lerobot/scripts/offline_eval_act_sequence.py \
      --checkpoint "$PROJECT/outputs/train/$train_dir/checkpoints/$step/pretrained_model" \
      --dataset-root "$data_root" --repo-id "$repo_id" --episodes "$EPISODES" \
      --output "$OUT/$label" --device cuda > "$OUT/logs/$label.log" 2>&1; then
      echo 0 > "$OUT/logs/$label.exit"
    else
      echo $? > "$OUT/logs/$label.exit"
      return 1
    fi
  done
}

run_family RESETPAIR F3_P3_DET_RESETPAIR_10K_s1000 \
  "$PROJECT/数据集/formal3/kind_merged_nomaster_reset_pair_fit48" \
  QYyyyyyyy/formal3_kind_merged_nomaster_reset_pair_fit48

printf 'exit=0\n' > "$OUT/evaluation.done"
