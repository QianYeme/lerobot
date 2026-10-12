#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PATH="/root/miniconda3/bin:$PATH"
ROOT="$PROJECT/formal4_data/kind_merged_nomaster"
REPO="formal4/kind_merged_nomaster"
CKPT="$PROJECT/results/experiments/T3_JOINT_STATE_SPATIAL_CONDITIONING/T3_B3_s1/checkpoints/100000/pretrained_model"
RES="$PROJECT/experiments/position/t3_joint_state_spatial/index1/result"
python scripts/eval_t3_joint_state_spatial.py --checkpoint "$CKPT" --repo-id "$REPO" --root "$ROOT" --output-dir "$RES/dev_eval/B3/100000" --episodes 3,5,12,15,21,28,33,39,41,47,53,58 --batch-size 4 --num-workers 4
python scripts/report_t3_joint_state_spatial.py --result-root "$RES"
echo "T3 B3 evaluation complete."
