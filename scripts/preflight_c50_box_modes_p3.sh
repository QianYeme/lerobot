#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/bin:$PATH"

DATA_ROOT="$PROJECT/数据集/formal1_C50_nomaster_fit32"
DATASET_REPO_ID="QYyyyyyyy/formal1_C50_nomaster_fit32"
RECORDS="outputs/c50_box_condition_modes_20260921/p3_short"
READY="outputs/formal1_C50_nomaster_phase1/preparation/fit32_preparation_ready.json"
mkdir -p "$RECORDS/preflight"
[[ -f "$READY" ]] || { echo "fit32 preparation Gate is missing" >&2; exit 1; }
[[ ! -e "$RECORDS/preflight.done" ]] || { echo "preflight already completed" >&2; exit 1; }

python scripts/check_box_mode_initialization.py \
  --dataset.repo_id "$DATASET_REPO_ID" --dataset.root "$DATA_ROOT" --seed 1000 \
  --output "$RECORDS/shared_initialization.json" \
  2>&1 | tee "$RECORDS/shared_initialization.log"

export DATA_ROOT
TRAIN_EPISODES="$(python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['DATA_ROOT'])
manifest = json.loads((root / 'meta/nomaster_manifest_fit32.json').read_text())
print(json.dumps(manifest['train']))
PY
)"

run_mode() {
  local label="$1"
  local mode="$2"
  local out="outputs/train/C50_NOMASTER_${label}_p3_preflight_s1000"
  [[ ! -e "$out" ]] || { echo "refusing to overwrite $out" >&2; return 1; }
  python -u -m lerobot.scripts.lerobot_train \
    --policy.type=act_det --policy.device=cuda --policy.push_to_hub=false --policy.use_amp=false \
    --dataset.repo_id="$DATASET_REPO_ID" --dataset.root="$DATA_ROOT" \
    --dataset.episodes="$TRAIN_EPISODES" --dataset.video_backend=pyav \
    --dataset.use_imagenet_stats=true --dataset.image_transforms.enable=false \
    --policy.chunk_size=100 --policy.n_action_steps=1 \
    --policy.gripper_loss_weight=3.0 --policy.optimizer_lr_backbone=0.0001 \
    --policy.use_detection=true --policy.use_mask_guidance=false \
    --policy.fcos_feature_inject=false --policy.mask_feature_inject=false \
    --policy.use_explicit_box_condition=true --policy.box_condition_mode="$mode" \
    --policy.box_condition_camera=observation.images.top \
    --policy.box_condition_score_threshold=0.25 \
    --policy.box_condition_dropout=0.0 --policy.box_condition_noise_std=0.0 \
    --policy.box_action_residual_alpha=0.05 \
    --policy.aug_enable=false --policy.det_weight=1.0 \
    --policy.annotation_dir="$DATA_ROOT/annotations" \
    '--policy.det_cameras={"observation.images.top":{"enable":true},"observation.images.gripper":{"enable":false}}' \
    --seed=1000 --batch_size=8 --num_workers=4 \
    --steps=2 --log_freq=1 --save_freq=1 --eval_freq=0 --wandb.enable=false \
    --output_dir="$out"
}

labels=(C0 C1 C2)
modes=(token state action_residual)
pids=()
for index in "${!labels[@]}"; do
  run_mode "${labels[$index]}" "${modes[$index]}" \
    > "$RECORDS/preflight/${labels[$index]}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo 0 > "$RECORDS/preflight/${labels[$index]}.exit"
  else
    status=$?
    echo "$status" > "$RECORDS/preflight/${labels[$index]}.exit"
    failed=1
  fi
done
[[ "$failed" -eq 0 ]] || { echo "at least one preflight training failed" >&2; exit 1; }

for label in "${labels[@]}"; do
  python scripts/smoke_c0_explicit_box.py \
    --checkpoint "outputs/train/C50_NOMASTER_${label}_p3_preflight_s1000/checkpoints/last/pretrained_model" \
    --dataset.repo_id QYyyyyyyy/formal1_C50_nomaster \
    --dataset.root "$PROJECT/数据集/formal1_C50_nomaster" \
    --episodes 33,10,6 --expected-state-dim 8 --device cuda \
    --output "$RECORDS/preflight/${label}_smoke.json" \
    > "$RECORDS/preflight/${label}_smoke.log" 2>&1
done

printf 'exit=0\n' > "$RECORDS/preflight.done"
date -Is > "$RECORDS/preflight_finished_at.txt"
