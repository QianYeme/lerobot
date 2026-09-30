#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
CAMPAIGN="$PROJECT/outputs/formal3_preroll_p2_20260929"

while [[ ! -f "$CAMPAIGN/train.done" ]]; do
  if [[ -f "$CAMPAIGN/train_DET_RESET_PREROLL_10K.exit" ]] &&
     [[ "$(cat "$CAMPAIGN/train_DET_RESET_PREROLL_10K.exit")" != "0" ]]; then
    echo "training failed; refusing evaluation" >&2
    exit 1
  fi
  sleep 30
done

bash scripts/run_formal3_reset_preroll_cross_eval_parallel.sh > "$CAMPAIGN/cross_eval_driver.log" 2>&1 &
p1=$!
bash scripts/run_formal3_reset_preroll_dev12_eval_parallel.sh > "$CAMPAIGN/dev12_eval_driver.log" 2>&1 &
p2=$!
failed=0
wait "$p1" || failed=1
wait "$p2" || failed=1
echo "$failed" > "$CAMPAIGN/posttrain_eval.exit"
[[ "$failed" -eq 0 ]]
printf 'exit=0\n' > "$CAMPAIGN/evaluation.done"
