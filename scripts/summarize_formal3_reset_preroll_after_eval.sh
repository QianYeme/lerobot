#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/bin:$PATH"

CAMPAIGN="$PROJECT/outputs/formal3_preroll_p2_20260929"
while [[ ! -f "$CAMPAIGN/evaluation.done" ]]; do
  if [[ -f "$CAMPAIGN/posttrain_eval.exit" ]] &&
     [[ "$(cat "$CAMPAIGN/posttrain_eval.exit")" != "0" ]]; then
    echo "evaluation failed; refusing summary" >&2
    exit 1
  fi
  sleep 30
done

python scripts/summarize_formal3_reset_pair_p3.py \
  --variant preroll \
  --campaign "$CAMPAIGN" \
  --timetrim-campaign "$PROJECT/outputs/formal3_time_trim_p3_20260928" \
  --resetpair-root "$PROJECT/formal3_data/kind_merged_nomaster_reset_preroll_fit48" \
  --source-root "$PROJECT/formal3_data/kind_merged_nomaster_reset_preroll_fit48" \
  --timetrim-root "$PROJECT/formal3_data/kind_merged_nomaster_time_trim_fit48" \
  --original-root "$PROJECT/formal3_data/kind_merged_nomaster_fit48"

printf 'exit=0\n' > "$CAMPAIGN/summary.done"
