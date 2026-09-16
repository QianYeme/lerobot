#!/usr/bin/env bash
# Run from an activated LeRobot environment; also works in a fresh screen shell.
set -euo pipefail

MODEL="${1:-all}"
STEPS="${2:-20000}"
case "$MODEL" in act|det|inject|all) ;; *) echo 'Usage: bash scripts/train_c50.sh [act|det|inject|all] [steps]' >&2; exit 2 ;; esac
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || { echo 'steps must be a positive integer' >&2; exit 2; }

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export DATA_ROOT="${DATA_ROOT:-$PROJECT/数据集/formal1_C}"
export PYTHONNOUSERSITE=1
export TRAIN_EPISODES
TRAIN_EPISODES="$(python - <<'PY'
import json
validation = {7, 11, 13, 14, 16, 27, 35, 38, 40, 43}
training = [ep for ep in range(50) if ep not in validation]
assert len(training) == 40 and not set(training) & validation
print(json.dumps(training))
PY
)"
[[ -f "$DATA_ROOT/meta/info.json" ]] || { echo "Missing dataset: $DATA_ROOT" >&2; exit 1; }
command -v lerobot-train >/dev/null

run_model () {
  local model="$1" policy_type=act name=C50_ACT_s1000
  local extra=()
  if [[ "$model" != act ]]; then
    policy_type=act_det
    name=C50_DET_s1000
    local inject=false
    if [[ "$model" == inject ]]; then
      inject=true
      name=C50_INJECT_s1000
    fi
    extra=(
      --policy.use_detection=true
      --policy.use_mask_guidance=false
      --policy.fcos_feature_inject="$inject"
      '--policy.fcos_inject_levels=["p4"]'
      --policy.mask_feature_inject=false
      --policy.aug_enable=false
      --policy.det_weight=1.0
      --policy.annotation_dir="$DATA_ROOT/annotations"
      '--policy.det_cameras={"observation.images.top":{"enable":true},"observation.images.gripper":{"enable":false}}'
    )
  fi
  if [[ "$STEPS" == 2000 ]]; then name="${name%_s1000}_smoke_s1000"; fi
  local out="outputs/train/$name"
  local records="outputs/formal1_C50_phase1/$name"
  if [[ -e "$out" || -e "$records" ]]; then
    echo "Existing run: $name. Inspect it and use checkpoint resume for continuation; nothing overwritten." >&2
    return 1
  fi
  mkdir -p "$records"
  git rev-parse HEAD > "$records/code_commit.txt"
  git diff HEAD -- src scripts > "$records/code_changes.patch"
  python -m pip freeze > "$records/environment.txt"
  python - <<'PY' > "$records/split.json"
import json, os
print(json.dumps({'train': json.loads(os.environ['TRAIN_EPISODES']),
                  'validation': [7,11,13,14,16,27,35,38,40,43]}, indent=2))
PY
  local cmd=(
    lerobot-train
    --policy.type="$policy_type" --policy.device=cuda --policy.push_to_hub=false
    --dataset.repo_id=QYyyyyyyy/formal1_C --dataset.root="$DATA_ROOT"
    --dataset.episodes="$TRAIN_EPISODES" --dataset.video_backend=pyav
    --dataset.use_imagenet_stats=true --dataset.image_transforms.enable=false
    --policy.chunk_size=100 --policy.n_action_steps=1
    --policy.gripper_loss_weight=3.0 --policy.optimizer_lr_backbone=0.0001
    --seed=1000 --batch_size=8 --num_workers=4
    --steps="$STEPS" --log_freq=100 --save_freq=2000
    --eval_freq=0 --wandb.enable=false --output_dir="$out"
    "${extra[@]}"
  )
  printf '%q ' "${cmd[@]}" > "$records/command.sh"
  printf '\n' >> "$records/command.sh"
  "${cmd[@]}" 2>&1 | tee "$records/train.log"
}

if [[ "$MODEL" == all ]]; then
  for model in act det inject; do run_model "$model"; done
else
  run_model "$MODEL"
fi
