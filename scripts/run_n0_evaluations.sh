#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PATH="/root/miniconda3/bin:$PATH"
ROOT="$PROJECT/formal4_data/kind_merged_nomaster"
REPO="formal4/kind_merged_nomaster"
BASE="$PROJECT/results/experiments/N0_OVERNIGHT_SPATIAL_CONDITIONING_BATCH"
RES="$PROJECT/experiments/position/n0_overnight_batch/index1/result"
for spec in \
  "R2_ACTION_LEVEL:$BASE/R2_ACTION_LEVEL_s1/checkpoints/100000/pretrained_model" \
  "R1_RELATIONAL:$BASE/R1_RELATIONAL_s1/checkpoints/100000/pretrained_model" \
  "R3_FILM:$BASE/R3_FILM_s1/checkpoints/100000/pretrained_model"; do
  IFS=: read -r label ckpt <<< "$spec"
  python scripts/eval_n0_candidate.py --candidate "$label" --checkpoint "$ckpt" --repo-id "$REPO" --root "$ROOT" --output-dir "$RES/$label/100000" --episodes 3,5,12,15,21,28,33,39,41,47,53,58 --batch-size 4 --num-workers 4
  python scripts/eval_t0_2_token_transplant.py --checkpoint "$ckpt" --repo-id "$REPO" --root "$ROOT" --episodes 3,5,12,15,21,28,33,39,41,47,53,58 --output-dir "$RES/$label/transplant" --allow-nonp1-checkpoint
done
python scripts/report_n0_overnight_batch.py --result-root "$RES"
echo "N0 evaluations complete."
