#!/usr/bin/env bash
# Retrain DET_BASE on the shoulder_lift-clipped formal3 dataset.
#
# Fix under validation (2026-09-28): the original models froze at the grasp
# because the leader/action shoulder_lift label overshoots the follower's
# reachable grasp depth. `prepare_formal3_clip_shoulder_lift.py` clips the
# action to the follower's observed state range.
#
# PRE-STEPS (run once on the server that holds the full data, in order):
#   1. python scripts/prepare_formal3_clip_shoulder_lift.py \
#        --source 数据集/formal3/kind_merged_nomaster \
#        --destination 数据集/formal3/kind_merged_nomaster_clipsh \
#        --output outputs/formal3_clipsh_20260928/clip_record.json
#   2. python scripts/prepare_formal3_fit48_dataset.py \
#        --source-root 数据集/formal3/kind_merged_nomaster_clipsh \
#        --target 数据集/formal3/kind_merged_nomaster_clipsh_fit48 \
#        --output outputs/formal3_clipsh_20260928/fit48_record.json
# Then run this script: `train_formal3_clipsh_retrain.sh preflight`, review,
# authorize, then `train_formal3_clipsh_retrain.sh train`.
#
# Validation-first: only DET_BASE is retrained here. If it recovers the grasp,
# extend to the other three models with the same flags as
# `train_formal3_mixed_4way.sh`.
set -euo pipefail

MODE="${1:-preflight}"
[[ "$MODE" == preflight || "$MODE" == train ]] || {
  echo 'usage: train_formal3_clipsh_retrain.sh [preflight|train]' >&2
  exit 2
}

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/bin:$PATH"

DATA_ROOT="$PROJECT/数据集/formal3/kind_merged_nomaster_clipsh_fit48"
SOURCE_ROOT="$PROJECT/数据集/formal3/kind_merged_nomaster_clipsh"
PHASE_LABELS="$PROJECT/outputs/formal3_phase_labels_20260927/final_reviewed_labels/reviewed_phase_labels.parquet"
CAMPAIGN="$PROJECT/outputs/formal3_clipsh_20260928"
SPLIT_READY="$PROJECT/outputs/formal3_clipsh_20260928/fit48_record.json"

[[ -f "$SPLIT_READY" ]] || { echo "missing clip+fit48 record: $SPLIT_READY" >&2; exit 1; }
python - "$SPLIT_READY" "$DATA_ROOT" "$SOURCE_ROOT" <<'PY'
import json, sys
from pathlib import Path
rec = json.loads(Path(sys.argv[1]).read_text())
root, source = Path(sys.argv[2]), Path(sys.argv[3])
assert rec["status"] == "passed"
manifest = json.loads((root / "meta/formal3_fit48_manifest.json").read_text())
assert len(manifest["train"]) == 48 and len(manifest["development"]) == 12
assert (source / "annotations/top/episode_000.xml").is_file()
PY

TRAIN_EPISODES="$(python - "$DATA_ROOT" <<'PY'
import json, sys
from pathlib import Path
print(json.dumps(json.loads((Path(sys.argv[1]) / "meta/formal3_fit48_manifest.json").read_text())["train"]))
PY
)"

if [[ "$MODE" == preflight ]]; then
  STEPS=2
  SAVE_FREQ=2
  OUT_ROOT="$CAMPAIGN/preflight"
else
  grep -q '状态：`AUTHORIZED`' "$CAMPAIGN/experiment.md" || {
    echo 'experiment is not authorized' >&2
    exit 1
  }
  STEPS=100000
  SAVE_FREQ=20000
  OUT_ROOT="$PROJECT/outputs/train"
fi
mkdir -p "$CAMPAIGN/logs" "$OUT_ROOT"

output="$OUT_ROOT/DET_BASE"
[[ "$MODE" == preflight ]] || output="$OUT_ROOT/F3_P4_DET_BASE_CLIPSH_s1000"
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 1; }

python -u -m lerobot.scripts.lerobot_train \
  --policy.type=act_det --policy.device=cuda --policy.push_to_hub=false --policy.use_amp=false \
  --dataset.repo_id=formal3/kind_merged_nomaster_clipsh_fit48 --dataset.root="$DATA_ROOT" \
  --dataset.episodes="$TRAIN_EPISODES" --dataset.video_backend=pyav \
  --dataset.use_imagenet_stats=true --dataset.image_transforms.enable=false \
  --policy.chunk_size=100 --policy.n_action_steps=1 --policy.gripper_loss_weight=3.0 \
  --policy.optimizer_lr_backbone=0.0001 --policy.use_detection=true --policy.use_mask_guidance=false \
  --policy.fcos_feature_inject=false --policy.mask_feature_inject=false \
  --policy.use_explicit_box_condition=false --policy.box_condition_mode=state \
  --policy.box_condition_camera=observation.images.top --policy.box_condition_score_threshold=0.25 \
  --policy.box_condition_dropout=0.0 --policy.box_condition_noise_std=0.0 \
  --policy.use_phase_aux=false --policy.phase_labels="$PHASE_LABELS" \
  --policy.phase_weight=0.0 --policy.phase_num_classes=4 \
  --policy.aug_enable=false --policy.det_weight=1.0 --policy.annotation_dir="$SOURCE_ROOT/annotations" \
  '--policy.det_cameras={"observation.images.top":{"enable":true},"observation.images.gripper":{"enable":false}}' \
  --seed=1000 --batch_size=8 --num_workers=4 --steps="$STEPS" --log_freq=1 \
  --save_freq="$SAVE_FREQ" --eval_freq=0 --wandb.enable=false --output_dir="$output" \
  2>&1 | tee "$CAMPAIGN/logs/${MODE}_DET_BASE.log"

printf 'exit=%s\n' "$?" > "$CAMPAIGN/logs/${MODE}_DET_BASE.exit"
