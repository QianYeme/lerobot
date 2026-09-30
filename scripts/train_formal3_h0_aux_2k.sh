#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/bin:$PATH"

DATA="$PROJECT/formal3_data/kind_merged_nomaster_reset_preroll_fit48"
ANNOTATIONS="$PROJECT/formal3_data/kind_merged_nomaster_reset_preroll/annotations"
CAMPAIGN="$PROJECT/outputs/formal3_h0_aux_p3_20260929"

python - "$CAMPAIGN/preflight/preflight.done" <<'PY'
import sys
from pathlib import Path
assert Path(sys.argv[1]).is_file(), "preflight.done missing; run preflight_formal3_h0_aux.sh first"
PY

TRAIN_EPISODES="$(python - "$DATA" <<'PY'
import json, sys
from pathlib import Path
print(json.dumps(json.loads((Path(sys.argv[1]) / "meta/formal3_fit48_manifest.json").read_text())["train"]))
PY
)"
mkdir -p "$CAMPAIGN/logs"

# Two parallel 2k candidate trainings (W=3 and W=10). Same config as the
# preroll 10k arm plus the h0 auxiliary flags; only the weight differs.
declare -A pids
for w in 3 10; do
  OUT="$PROJECT/outputs/train/F3_P3_DET_H0AUX_W0${w}_2K_s1000"
  [[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 1; }
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
    --policy.use_h0_action_aux=true --policy.h0_action_weight="$w" \
    --seed=1000 --batch_size=8 --num_workers=4 --steps=2000 --log_freq=100 \
    --save_freq=2000 --eval_freq=0 --wandb.enable=false --output_dir="$OUT" \
    > "$CAMPAIGN/logs/train_H0AUX_W0${w}_2K.log" 2>&1 &
  pids[$w]=$!
done

failed=0
for w in "${!pids[@]}"; do
  if wait "${pids[$w]}"; then code=0; else code=$?; failed=1; fi
  echo "$code" > "$CAMPAIGN/logs/train_H0AUX_W0${w}_2K.exit"
done

for w in 3 10; do
  log="$CAMPAIGN/logs/train_H0AUX_W0${w}_2K.log"
  out="$PROJECT/outputs/train/F3_P3_DET_H0AUX_W0${w}_2K_s1000"
  [[ -d "$out/checkpoints/002000/pretrained_model" ]] || { echo "W${w}: step-2000 checkpoint missing" >&2; exit 1; }
  if grep -Eqi '(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)|out of memory|traceback' "$log"; then
    echo "W${w}: training log contains a failure marker" >&2
    exit 1
  fi
done

[[ "$failed" -eq 0 ]] || exit 1
printf 'exit=0\n' > "$CAMPAIGN/train.done"
date -Is > "$CAMPAIGN/finished_at.txt"
