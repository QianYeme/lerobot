#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/bin:$PATH"

DATA="$PROJECT/formal3_data/kind_merged_nomaster_reset_preroll_fit48"
ANNOTATIONS="$PROJECT/formal3_data/kind_merged_nomaster_reset_preroll/annotations"
CAMPAIGN="$PROJECT/outputs/formal3_preroll_p2_20260929"
OUTPUT="$PROJECT/outputs/train/F3_P3_DET_RESET_PREROLL_10K_s1000"
LOG="$CAMPAIGN/train_DET_RESET_PREROLL_10K.log"

[[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite $OUTPUT" >&2; exit 1; }
python - "$CAMPAIGN/preflight_ready.json" <<'PY'
import json, sys
ready = json.load(open(sys.argv[1]))
assert ready["status"] == "passed"
assert ready["remote_2step"]["exit"] == 0
assert ready["fit48"]["train_frames"] == 19198
PY

TRAIN_EPISODES="$(python - "$DATA" <<'PY'
import json, sys
from pathlib import Path
print(json.dumps(json.loads((Path(sys.argv[1]) / "meta/formal3_fit48_manifest.json").read_text())["train"]))
PY
)"
mkdir -p "$(dirname "$OUTPUT")" "$CAMPAIGN"

python -u -m lerobot.scripts.lerobot_train \
  --policy.type=act_det --policy.device=cuda --policy.push_to_hub=false --policy.use_amp=false \
  --dataset.repo_id=QYyyyyyyy/formal3_kind_merged_nomaster_reset_preroll_fit48 \
  --dataset.root="$DATA" --dataset.episodes="$TRAIN_EPISODES" --dataset.video_backend=pyav \
  --dataset.use_imagenet_stats=true --dataset.image_transforms.enable=false \
  --policy.chunk_size=100 --policy.n_action_steps=1 --policy.gripper_loss_weight=3.0 \
  --policy.optimizer_lr_backbone=0.0001 --policy.use_detection=true \
  --policy.use_mask_guidance=false --policy.fcos_feature_inject=false \
  --policy.mask_feature_inject=false --policy.use_explicit_box_condition=false \
  --policy.use_phase_aux=false --policy.aug_enable=false \
  --policy.det_weight=1.0 --policy.annotation_dir="$ANNOTATIONS" \
  '--policy.det_cameras={"observation.images.top":{"enable":true},"observation.images.gripper":{"enable":false}}' \
  --seed=1000 --batch_size=8 --num_workers=4 --steps=10000 --log_freq=100 \
  --save_freq=2000 --eval_freq=0 --wandb.enable=false --output_dir="$OUTPUT" \
  > "$LOG" 2>&1

code=$?
echo "$code" > "$CAMPAIGN/train_DET_RESET_PREROLL_10K.exit"
[[ "$code" -eq 0 ]] || exit "$code"
if grep -Eqi '(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)|out of memory|traceback' "$LOG"; then
  echo "training log contains a failure marker" >&2
  exit 1
fi
printf 'exit=0\n' > "$CAMPAIGN/train.done"
date -Is > "$CAMPAIGN/finished_at.txt"
