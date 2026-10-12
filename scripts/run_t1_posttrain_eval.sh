#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/bin:$PATH"
ROOT="$PROJECT/formal4_data/kind_merged_nomaster"
COMMON=(--repo-id formal4/kind_merged_nomaster --root "$ROOT" --episodes 3,5,12,15,21,28,33,39,41,47,53,58 --batch-size 8 --num-workers 4)
while [[ ! -f experiments/position/t1_explicit_spatial_conditioning/index1/result/runs/T1_B1_s1/train.done || ! -f experiments/position/t1_explicit_spatial_conditioning/index1/result/runs/T1_P2_s1/train.done ]]; do
  sleep 60
done
eval_model() {
  local model="$1"
  local policy_dir="results/experiments/T1_OCACT_EXPLICIT_SPATIAL_CONDITIONING/T1_${model}_s1/checkpoints"
  local out_root="experiments/position/t1_explicit_spatial_conditioning/index1/result/dev_eval/${model}"
  mkdir -p "$out_root"
  for step in 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000; do
    local ckpt="$policy_dir/$step/pretrained_model"
    [[ -f "$ckpt/config.json" ]] || continue
    python -u scripts/eval_t1_spatial_mechanism.py --checkpoint "$ckpt" "${COMMON[@]}" --output-dir "$out_root/$step" > "$out_root/${step}.log" 2>&1
  done
}
eval_model B1 & p1=$!
eval_model P2 & p2=$!
wait "$p1" "$p2"
python scripts/report_t1_results.py --result-root experiments/position/t1_explicit_spatial_conditioning/index1/result
P2_BEST="$(python - <<'PY'
import json
p='experiments/position/t1_explicit_spatial_conditioning/index1/result/checkpoints.json'
print(json.load(open(p))['P2']['best']['checkpoint'])
PY
)"
mkdir -p experiments/position/t1_explicit_spatial_conditioning/index1/result/transplant_P2
python -u scripts/eval_t0_2_token_transplant.py --allow-nonp1-checkpoint --checkpoint "$P2_BEST" --repo-id formal4 --root "$ROOT" --episodes 3,5,12,15,21,28,33,39,41,47,53,58 --output-dir experiments/position/t1_explicit_spatial_conditioning/index1/result/transplant_P2 --batch-size 8 --num-workers 4 > experiments/position/t1_explicit_spatial_conditioning/index1/result/transplant_P2/run.log 2>&1
cp experiments/position/t1_explicit_spatial_conditioning/index1/result/transplant_P2/transplant_results.json experiments/position/t1_explicit_spatial_conditioning/index1/result/transplant_results.json
printf 'post_eval_exit=0\n' > experiments/position/t1_explicit_spatial_conditioning/index1/result/post_eval.done
