#!/usr/bin/env bash
set -euo pipefail

# B1 post-eval summarizer: waits for the posttrain evaluation families to
# complete, then computes the preregistered 100k gates
# (summarize_formal3_preroll_100k.py) and marks summary.done. Refuses to
# summarize if evaluation failed.

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/bin:$PATH"

CAMPAIGN="$PROJECT/outputs/formal3_preroll_p4_100k_20260929"
while [[ ! -f "$CAMPAIGN/evaluation.done" ]]; do
  if [[ -f "$CAMPAIGN/posttrain_eval.exit" ]] &&
     [[ "$(cat "$CAMPAIGN/posttrain_eval.exit")" != "0" ]]; then
    echo "evaluation failed; refusing summary" >&2
    exit 1
  fi
  sleep 30
done

python scripts/summarize_formal3_preroll_100k.py \
  --campaign "$CAMPAIGN" \
  --timetrim-campaign "$PROJECT/outputs/formal3_time_trim_p3_20260928" \
  --preroll-root "$PROJECT/formal3_data/kind_merged_nomaster_reset_preroll_fit48" \
  --timetrim-root "$PROJECT/formal3_data/kind_merged_nomaster_time_trim_fit48" \
  --original-root "$PROJECT/formal3_data/kind_merged_nomaster_fit48"

printf 'exit=0\n' > "$CAMPAIGN/summary.done"
