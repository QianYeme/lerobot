#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"

DATA_ROOT="${DATA_ROOT:-$PROJECT/数据集/formal1_C50_nomaster_fit32}"
REPO_ID="QYyyyyyyy/formal1_C50_nomaster_fit32"
EPISODES="7,11,13,14,16,27,35,38,40,43"
OUT="outputs/c50_c1_diffusion_p4_20260921/trajectory_val10"
mkdir -p "$OUT/logs"
[[ ! -e "$OUT/evaluation.done" ]] || { echo "trajectory evaluation already completed" >&2; exit 1; }

run_one() {
  local name="$1" checkpoint="$2"
  python -u src/lerobot/scripts/offline_eval_act_sequence.py \
    --checkpoint "$checkpoint" --dataset-root "$DATA_ROOT" --repo-id "$REPO_ID" \
    --episodes "$EPISODES" --output "$OUT/$name" --device cuda \
    > "$OUT/logs/$name.log" 2>&1
  echo 0 > "$OUT/logs/$name.exit"
}

run_one C1_BASE outputs/train/C50_P4_C1_BASE_s1000/checkpoints/100000/pretrained_model & p1=$!
run_one C1_DROP outputs/train/C50_P4_C1_DROP_s1000/checkpoints/100000/pretrained_model & p2=$!
run_one C1_NOISE outputs/train/C50_P4_C1_NOISE_s1000/checkpoints/100000/pretrained_model & p3=$!
run_one DIFFUSION_NM outputs/train/C50_P4_DIFFUSION_NM_s1000/checkpoints/100000/pretrained_model & p4=$!

failed=0
for pair in "C1_BASE:$p1" "C1_DROP:$p2" "C1_NOISE:$p3" "DIFFUSION_NM:$p4"; do
  name="${pair%%:*}"
  pid="${pair##*:}"
  if ! wait "$pid"; then
    echo 1 > "$OUT/logs/$name.exit"
    failed=1
  fi
done
[[ "$failed" -eq 0 ]] || { echo "at least one trajectory evaluation failed" >&2; exit 1; }
printf 'exit=0\n' > "$OUT/evaluation.done"
