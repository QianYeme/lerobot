#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/bin:$PATH"

DATA="$PROJECT/formal3_data/kind_merged_nomaster_reset_preroll_fit48"
ORIGINAL="$PROJECT/formal3_data/kind_merged_nomaster_fit48"
TIMETRIM="$PROJECT/formal3_data/kind_merged_nomaster_time_trim_fit48"
CAMPAIGN="$PROJECT/outputs/formal3_h0_aux_p3_20260929"
PREROLL_CAMPAIGN="$PROJECT/outputs/formal3_preroll_p2_20260929"
OUT="$CAMPAIGN/diagnose_train48"
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 1; }
mkdir -p "$OUT/logs"

python - "$CAMPAIGN/train.done" <<'PY'
import sys
from pathlib import Path
assert Path(sys.argv[1]).is_file(), "train.done missing; run train_formal3_h0_aux_2k.sh first"
PY

EPISODES="$(python - "$DATA" <<'PY'
import json, sys
from pathlib import Path
items = json.loads((Path(sys.argv[1]) / "meta/formal3_fit48_manifest.json").read_text())["train"]
print(",".join(map(str, items)))
PY
)"

# Parallel train48 frame0 evaluation for both 2k checkpoints.
declare -A pids
for w in 3 10; do
  label="H0AUX_W0${w}_002000"
  python -u src/lerobot/scripts/offline_eval_act_sequence.py \
    --checkpoint "$PROJECT/outputs/train/F3_P3_DET_H0AUX_W0${w}_2K_s1000/checkpoints/002000/pretrained_model" \
    --dataset-root "$DATA" --repo-id QYyyyyyyy/formal3_kind_merged_nomaster_reset_preroll_fit48 \
    --episodes "$EPISODES" --max-frames 2 --output "$OUT/$label" --device cuda \
    > "$OUT/logs/$label.log" 2>&1 &
  pids[$w]=$!
done

failed=0
for w in "${!pids[@]}"; do
  if wait "${pids[$w]}"; then code=0; else code=$?; failed=1; fi
  echo "$code" > "$OUT/logs/H0AUX_W0${w}_002000.exit"
done
[[ "$failed" -eq 0 ]] || exit 1

# Bias analysis on the two candidates and on the preroll-2k baseline
# (already evaluated on train48 frame0; no new evaluation needed).
for tag in "H0AUX_W03_002000:$OUT/H0AUX_W03_002000" \
           "H0AUX_W10_002000:$OUT/H0AUX_W10_002000" \
           "PREROLL_BASE_002000:$PREROLL_CAMPAIGN/diagnose_train48_reset/RESET_PREROLL_TRAIN48_002000"; do
  name="${tag%%:*}"
  dir="${tag#*:}"
  python scripts/analyze_formal3_h0_bias.py \
    --npz-dir "$dir" --original-root "$ORIGINAL" --timetrim-root "$TIMETRIM" \
    --episodes "$EPISODES" --out "$OUT/h0_bias_${name}.json"
done

printf 'exit=0\n' > "$OUT/diagnosis.done"
