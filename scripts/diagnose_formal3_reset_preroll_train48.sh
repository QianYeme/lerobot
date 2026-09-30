#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/bin:$PATH"

DATA="$PROJECT/formal3_data/kind_merged_nomaster_reset_preroll_fit48"
CAMPAIGN="$PROJECT/outputs/formal3_preroll_p2_20260929"
OUT="$CAMPAIGN/diagnose_train48_reset"
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 1; }
mkdir -p "$OUT/logs"

EPISODES="$(python - "$DATA" <<'PY'
import json, sys
from pathlib import Path
items = json.loads((Path(sys.argv[1]) / "meta/formal3_fit48_manifest.json").read_text())["train"]
print(",".join(map(str, items)))
PY
)"

declare -A pids
for step in 002000 008000 010000; do
  label="RESET_PREROLL_TRAIN48_${step}"
  python -u src/lerobot/scripts/offline_eval_act_sequence.py \
    --checkpoint "$PROJECT/outputs/train/F3_P3_DET_RESET_PREROLL_10K_s1000/checkpoints/$step/pretrained_model" \
    --dataset-root "$DATA" --repo-id QYyyyyyyy/formal3_kind_merged_nomaster_reset_preroll_fit48 \
    --episodes "$EPISODES" --max-frames 2 --output "$OUT/$label" --device cuda \
    > "$OUT/logs/$label.log" 2>&1 &
  pids[$step]=$!
done

failed=0
for step in "${!pids[@]}"; do
  if wait "${pids[$step]}"; then code=0; else code=$?; failed=1; fi
  echo "$code" > "$OUT/logs/RESET_PREROLL_TRAIN48_${step}.exit"
done
[[ "$failed" -eq 0 ]] || exit 1
printf 'exit=0\n' > "$OUT/evaluation.done"
