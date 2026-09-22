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
PREFLIGHT_RECORDS="outputs/c50_c1_diffusion_p4_20260921_v2"
[[ -f "$PREFLIGHT_RECORDS/preflight.done" ]] || { echo "preflight Gate is missing" >&2; exit 1; }
grep -q '`READY`' "$RECORDS/experiment.md" || { echo "experiment contract is not READY" >&2; exit 1; }
[[ ! -e "$RECORDS/train.done" ]] || { echo "training already completed" >&2; exit 1; }
mkdir -p "$RECORDS/logs"
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
    --seed=1000 --batch_size=8 --num_workers=4 --steps=100000 --log_freq=100 --save_freq=20000 \
    --eval_freq=0 --wandb.enable=false --output_dir="outputs/train/C50_P4_${label}_s1000"
}

run_diffusion() {
  python -u -m lerobot.scripts.lerobot_train \
    --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false --policy.use_amp=false \
    --dataset.repo_id="$DATASET_REPO_ID" --dataset.root="$DATA_ROOT" --dataset.episodes="$TRAIN_EPISODES" \
    --dataset.video_backend=pyav --dataset.use_imagenet_stats=true --dataset.image_transforms.enable=false \
    --policy.horizon=16 --policy.n_obs_steps=1 --policy.n_action_steps=1 --policy.num_inference_steps=10 \
    --policy.drop_n_last_frames=0 --policy.vision_backbone=resnet18 --seed=1000 --batch_size=8 --num_workers=4 \
    --steps=100000 --log_freq=100 --save_freq=20000 --eval_freq=0 --wandb.enable=false \
    --output_dir=outputs/train/C50_P4_DIFFUSION_NM_s1000
}

run_c1 C1_BASE 0.0 0.0 > "$RECORDS/logs/C1_BASE.log" 2>&1 & p1=$!
run_c1 C1_DROP 0.10 0.0 > "$RECORDS/logs/C1_DROP.log" 2>&1 & p2=$!
run_c1 C1_NOISE 0.0 0.03 > "$RECORDS/logs/C1_NOISE.log" 2>&1 & p3=$!
run_diffusion > "$RECORDS/logs/DIFFUSION_NM.log" 2>&1 & p4=$!

failed=0
for pair in "C1_BASE:$p1" "C1_DROP:$p2" "C1_NOISE:$p3" "DIFFUSION_NM:$p4"; do
  label="${pair%%:*}"; pid="${pair##*:}"
  if wait "$pid"; then echo 0 > "$RECORDS/logs/${label}.exit"; else echo $? > "$RECORDS/logs/${label}.exit"; failed=1; fi
done
[[ "$failed" -eq 0 ]] || { echo "at least one P4 training run failed" >&2; exit 1; }
printf 'exit=0\n' > "$RECORDS/train.done"
date -Is > "$RECORDS/finished_at.txt"
