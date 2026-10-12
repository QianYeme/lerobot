#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?Usage: eval_t2_fixed_scale.sh b2|p3}"
STEP="${2:-100000}"
case "$MODEL" in b2|p3) ;; *) exit 2 ;; esac
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/bin:$PATH"
DATA_ROOT="$PROJECT/formal4_data/kind_merged_nomaster"
REPO_ID="formal4/kind_merged_nomaster"
CKPT="$PROJECT/results/experiments/T2_FIXED_SCALE_SPATIAL_CONDITIONING/T2_${MODEL^^}_s1/checkpoints/${STEP}/pretrained_model"
OUT="$PROJECT/experiments/position/t2_fixed_scale_spatial/index1/result/dev_eval/${MODEL^^}/${STEP}"
[[ -d "$CKPT" ]] || { echo "missing checkpoint: $CKPT" >&2; exit 1; }
mkdir -p "$OUT"
python -u scripts/eval_t1_spatial_mechanism.py \
  --checkpoint "$CKPT" \
  --repo-id "$REPO_ID" \
  --root "$DATA_ROOT" \
  --episodes 3,5,12,15,21,28,33,39,41,47,53,58 \
  --output-dir "$OUT" \
  --batch-size 8 --num-workers 4
python - <<PY
import json
from pathlib import Path
p=Path("$OUT")
m=json.loads((p/"metrics.json").read_text())
m["experiment"]="T2_FIXED_SCALE_SPATIAL_CONDITIONING"
m["model"]="${MODEL^^}"
m["fixed_scale"] = 0.5 if "${MODEL}" == "p3" else None
m["t1_evaluator_compat"] = True
(p/"metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
v=json.loads((p/"validation.json").read_text())
v.update({"experiment":"T2_FIXED_SCALE_SPATIAL_CONDITIONING", "model":"${MODEL^^}", "fixed_scale":m["fixed_scale"], "final_test_used":False, "real_robot_used":False})
(p/"validation.json").write_text(json.dumps(v, indent=2), encoding="utf-8")
PY
