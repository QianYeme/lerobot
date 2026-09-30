#!/usr/bin/env bash
set -euo pipefail

# B1 trend check at the 20k checkpoint (the reversal interval observed in
# prior campaigns): waits for the checkpoint to land, then runs a lightweight
# frame0 eval (dev12 + train48, max-frames 2) plus the h0 bias analysis into
# trend_20k/. Evidence only -- no promotion decisions are made here, and the
# posttrain eval dirs (eval_dev12/eval_cross_on_timetrim/eval_train48) are
# left untouched for the official watcher.

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/bin:$PATH"

CAMPAIGN="$PROJECT/outputs/formal3_preroll_p4_100k_20260929"
CKPT="$PROJECT/outputs/train/F3_P4_DET_RESET_PREROLL_100K_s1000/checkpoints/020000/pretrained_model"
DATA="$PROJECT/formal3_data/kind_merged_nomaster_reset_preroll_fit48"
OUT="$CAMPAIGN/trend_20k"

while [[ ! -e "$CKPT" ]]; do
  if [[ -f "$CAMPAIGN/train_DET_RESET_PREROLL_100K.exit" ]] &&
     [[ "$(cat "$CAMPAIGN/train_DET_RESET_PREROLL_100K.exit")" != "0" ]]; then
    echo "training failed; refusing trend check" >&2
    exit 1
  fi
  sleep 60
done
sleep 30  # let the checkpoint finish flushing

[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 1; }
mkdir -p "$OUT/logs"

DEV12_EPISODES="9,11,17,25,26,28,30,31,43,52,55,59"
TRAIN48_EPISODES="$(python - "$DATA" <<'PY'
import json, sys
from pathlib import Path
items = json.loads((Path(sys.argv[1]) / "meta/formal3_fit48_manifest.json").read_text())["train"]
print(",".join(map(str, items)))
PY
)"

python -u src/lerobot/scripts/offline_eval_act_sequence.py \
  --checkpoint "$CKPT" --dataset-root "$DATA" \
  --repo-id QYyyyyyyy/formal3_kind_merged_nomaster_reset_preroll_fit48 \
  --episodes "$DEV12_EPISODES" --max-frames 2 --output "$OUT/dev12_020000" --device cuda \
  > "$OUT/logs/dev12_020000.log" 2>&1 &
p1=$!
python -u src/lerobot/scripts/offline_eval_act_sequence.py \
  --checkpoint "$CKPT" --dataset-root "$DATA" \
  --repo-id QYyyyyyyy/formal3_kind_merged_nomaster_reset_preroll_fit48 \
  --episodes "$TRAIN48_EPISODES" --max-frames 2 --output "$OUT/train48_020000" --device cuda \
  > "$OUT/logs/train48_020000.log" 2>&1 &
p2=$!

wait "$p1"; c1=$?
wait "$p2"; c2=$?
echo "$c1" > "$OUT/logs/dev12_020000.exit"
echo "$c2" > "$OUT/logs/train48_020000.exit"
[[ "$c1" -eq 0 && "$c2" -eq 0 ]] || { echo "trend evals failed" >&2; exit 1; }

python scripts/analyze_formal3_h0_bias.py \
  --npz-dir "$OUT/dev12_020000" \
  --original-root "$PROJECT/formal3_data/kind_merged_nomaster_fit48" \
  --timetrim-root "$PROJECT/formal3_data/kind_merged_nomaster_time_trim_fit48" \
  --episodes "$DEV12_EPISODES" --out "$OUT/dev12_020000_bias.json"

python scripts/analyze_formal3_h0_bias.py \
  --npz-dir "$OUT/train48_020000" \
  --original-root "$PROJECT/formal3_data/kind_merged_nomaster_fit48" \
  --timetrim-root "$PROJECT/formal3_data/kind_merged_nomaster_time_trim_fit48" \
  --episodes "$TRAIN48_EPISODES" --out "$OUT/train48_020000_bias.json"

printf 'exit=0\n' > "$OUT/trend.done"
