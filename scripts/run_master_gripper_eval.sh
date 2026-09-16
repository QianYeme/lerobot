#!/usr/bin/env bash
# Run in the activated training environment; models run sequentially.
set -euo pipefail

MODEL="${1:-all}"
BATCH_SIZE="${2:-8}"
case "$MODEL" in act|det|inject|all) ;; *) echo 'Usage: bash scripts/run_master_gripper_eval.sh [act|det|inject|all] [batch_size]' >&2; exit 2 ;; esac
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo 'batch_size must be positive' >&2; exit 2; }
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-$PROJECT/数据集/formal1_C}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT/outputs/formal1_C50_phase1/master_gripper_ablation_100k}"
PHASE_FILE="${PHASE_FILE:-$OUTPUT_ROOT/phase_review/phase_annotations.json}"
[[ -f "$DATA_ROOT/meta/info.json" ]] || { echo "Missing dataset: $DATA_ROOT" >&2; exit 1; }
[[ -f "$PHASE_FILE" ]] || { echo "Missing phase annotations: $PHASE_FILE; run prepare_c50_phase_review.py first" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT"

run_model () {
  local name="$1"
  local output="$OUTPUT_ROOT/${name}_h0_master_zero.json"
  [[ ! -e "$output" && ! -e "${output%.json}.csv" && ! -e "${output%.json}.log" ]] || {
    echo "Existing artifacts for $name; use a different OUTPUT_ROOT (do not overwrite results)." >&2
    return 1
  }
  git rev-parse HEAD > "$OUTPUT_ROOT/${name}_code_commit.txt"
  git diff HEAD -- src/lerobot/scripts/eval_master_gripper_ablation.py scripts/prepare_c50_phase_review.py scripts/run_master_gripper_eval.sh > "$OUTPUT_ROOT/${name}_tracked_changes.patch"
  # New, untracked evaluator files are not included by git diff; archive exact sources.
  cp src/lerobot/scripts/eval_master_gripper_ablation.py "$OUTPUT_ROOT/${name}_evaluator.py"
  local cmd=(
    "$PYTHON_BIN" -u -m lerobot.scripts.eval_master_gripper_ablation
    --checkpoint "outputs/train/$name/checkpoints/100000/pretrained_model"
    --dataset.repo_id QYyyyyyyy/formal1_C --dataset.root "$DATA_ROOT"
    --episodes 7,11,13,14,16,27,35,38,40,43
    --phase-file "$PHASE_FILE" --batch-size "$BATCH_SIZE" --output "$output"
  )
  printf '%q ' "${cmd[@]}" > "$OUTPUT_ROOT/${name}_command.sh"
  printf '\n' >> "$OUTPUT_ROOT/${name}_command.sh"
  "${cmd[@]}" 2>&1 | tee "${output%.json}.log"
  printf 'exit=0\n' > "$OUTPUT_ROOT/${name}.done"
}

if [[ "$MODEL" == all ]]; then
  run_model C50_ACT_s1000
  run_model C50_DET_s1000
  run_model C50_INJECT_s1000
  printf 'all_models_complete\n' > "$OUTPUT_ROOT/complete.done"
else
  case "$MODEL" in
    act) run_model C50_ACT_s1000 ;;
    det) run_model C50_DET_s1000 ;;
    inject) run_model C50_INJECT_s1000 ;;
  esac
fi
