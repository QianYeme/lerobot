#!/usr/bin/env bash
set -euo pipefail

# B1 post-train watcher: wait for train.done, then run the five-checkpoint
# dev12 + time-trim cross evals in parallel, then the train48 eval, and mark
# evaluation.done. Refuses to evaluate if training failed.

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
CAMPAIGN="$PROJECT/outputs/formal3_preroll_p4_100k_20260929"

while [[ ! -f "$CAMPAIGN/train.done" ]]; do
  if [[ -f "$CAMPAIGN/train_DET_RESET_PREROLL_100K.exit" ]] &&
     [[ "$(cat "$CAMPAIGN/train_DET_RESET_PREROLL_100K.exit")" != "0" ]]; then
    echo "training failed; refusing evaluation" >&2
    exit 1
  fi
  sleep 30
done

bash scripts/run_formal3_reset_preroll_100k_cross_eval_parallel.sh > "$CAMPAIGN/cross_eval_driver.log" 2>&1 &
p1=$!
bash scripts/run_formal3_reset_preroll_100k_dev12_eval_parallel.sh > "$CAMPAIGN/dev12_eval_driver.log" 2>&1 &
p2=$!
failed=0
wait "$p1" || failed=1
wait "$p2" || failed=1
if [[ "$failed" -eq 0 ]]; then
  bash scripts/run_formal3_reset_preroll_100k_train48_eval_parallel.sh > "$CAMPAIGN/train48_eval_driver.log" 2>&1 || failed=1
fi
echo "$failed" > "$CAMPAIGN/posttrain_eval.exit"
[[ "$failed" -eq 0 ]]
printf 'exit=0\n' > "$CAMPAIGN/evaluation.done"
