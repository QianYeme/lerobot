#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/bin:$PATH"

DATA_ROOT="$PROJECT/数据集/formal1_C50_nomaster_fit32"
DATASET_REPO_ID="QYyyyyyyy/formal1_C50_nomaster_fit32"
RECORDS="outputs/c50_c1_diffusion_p4_20260921"
[[ ! -e "$RECORDS/preflight.done" ]] || { echo "preflight already completed" >&2; exit 1; }
mkdir -p "$RECORDS/preflight"
export DATA_ROOT
TRAIN_EPISODES="$(python - <<'PY'
import json, os
from pathlib import Path
print(json.dumps(json.loads((Path(os.environ['DATA_ROOT']) / 'meta/nomaster_manifest_fit32.json').read_text())['train']))
PY
)"

run_c1() {
  local label="$1" dropout="$2" noise="$3"
  python -u -m lerobot.scripts.lerobot_train \
    --policy.type=act_det --policy.device=cuda --policy.push_to_hub=false --policy.use_amp=false \
    --dataset.repo_id="$DATASET_REPO_ID" --dataset.root="$DATA_ROOT" --dataset.episodes="$TRAIN_EPISODES" \
    --dataset.video_backend=pyav --dataset.use_imagenet_stats=true --dataset.image_transforms.enable=false \
    --policy.chunk_size=100 --policy.n_action_steps=1 --policy.gripper_loss_weight=3.0 \
    --policy.optimizer_lr_backbone=0.0001 --policy.use_detection=true --policy.use_mask_guidance=false \
    --policy.fcos_feature_inject=false --policy.mask_feature_inject=false --policy.use_explicit_box_condition=true \
    --policy.box_condition_mode=state --policy.box_condition_camera=observation.images.top \
    --policy.box_condition_score_threshold=0.25 --policy.box_condition_dropout="$dropout" \
    --policy.box_condition_noise_std="$noise" --policy.box_action_residual_alpha=0.05 \
    --policy.aug_enable=false --policy.det_weight=1.0 --policy.annotation_dir="$DATA_ROOT/annotations" \
    '--policy.det_cameras={"observation.images.top":{"enable":true},"observation.images.gripper":{"enable":false}}' \
    --seed=1000 --batch_size=8 --num_workers=4 --steps=2 --log_freq=1 --save_checkpoint=false \
    --eval_freq=0 --wandb.enable=false --output_dir="$RECORDS/preflight/${label}"
}

run_diffusion() {
  python -u -m lerobot.scripts.lerobot_train \
    --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false --policy.use_amp=false \
    --dataset.repo_id="$DATASET_REPO_ID" --dataset.root="$DATA_ROOT" --dataset.episodes="$TRAIN_EPISODES" \
    --dataset.video_backend=pyav --dataset.use_imagenet_stats=true --dataset.image_transforms.enable=false \
    --policy.horizon=16 --policy.n_obs_steps=1 --policy.n_action_steps=1 --policy.num_inference_steps=10 \
    --policy.drop_n_last_frames=0 \
    --policy.vision_backbone=resnet18 --seed=1000 --batch_size=8 --num_workers=4 --steps=2 --log_freq=1 \
    --save_checkpoint=false --eval_freq=0 --wandb.enable=false --output_dir="$RECORDS/preflight/DIFFUSION_NM"
}

run_c1 C1_BASE 0.0 0.0 > "$RECORDS/preflight/C1_BASE.log" 2>&1 & p1=$!
run_c1 C1_DROP 0.10 0.0 > "$RECORDS/preflight/C1_DROP.log" 2>&1 & p2=$!
run_c1 C1_NOISE 0.0 0.03 > "$RECORDS/preflight/C1_NOISE.log" 2>&1 & p3=$!
run_diffusion > "$RECORDS/preflight/DIFFUSION_NM.log" 2>&1 & p4=$!

failed=0
for pair in "C1_BASE:$p1" "C1_DROP:$p2" "C1_NOISE:$p3" "DIFFUSION_NM:$p4"; do
  label="${pair%%:*}"; pid="${pair##*:}"
  if wait "$pid"; then echo 0 > "$RECORDS/preflight/${label}.exit"; else echo $? > "$RECORDS/preflight/${label}.exit"; failed=1; fi
done
[[ "$failed" -eq 0 ]] || { echo "preflight failed" >&2; exit 1; }
printf 'exit=0\n' > "$RECORDS/preflight.done"
