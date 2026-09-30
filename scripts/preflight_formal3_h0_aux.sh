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
OUT="$CAMPAIGN/preflight"
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 1; }
mkdir -p "$OUT/logs"

TRAIN_EPISODES="$(python - "$DATA" <<'PY'
import json, sys
from pathlib import Path
print(json.dumps(json.loads((Path(sys.argv[1]) / "meta/formal3_fit48_manifest.json").read_text())["train"]))
PY
)"

# Two parallel 2-step real-data preflights (W=3 and W=10), identical config to
# the preroll 10k arm plus the h0 auxiliary flags.
declare -A pids
for w in 3 10; do
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
    --seed=1000 --batch_size=8 --num_workers=4 --steps=2 --log_freq=1 \
    --save_freq=2 --eval_freq=0 --wandb.enable=false --output_dir="$OUT/W${w}" \
    > "$OUT/logs/W${w}.log" 2>&1 &
  pids[$w]=$!
done

failed=0
for w in "${!pids[@]}"; do
  if wait "${pids[$w]}"; then code=0; else code=$?; failed=1; fi
  echo "$code" > "$OUT/logs/W${w}.exit"
done

for w in 3 10; do
  log="$OUT/logs/W${w}.log"
  [[ -d "$OUT/W${w}/checkpoints/000002/pretrained_model" ]] || { echo "W${w}: step-2 checkpoint missing" >&2; exit 1; }
  if grep -Eqi '(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)|out of memory|traceback' "$log"; then
    echo "W${w}: log contains a failure marker" >&2
    exit 1
  fi
  # The h0 flags must be recorded in the saved policy config.
  python - "$OUT/W${w}/checkpoints/000002/pretrained_model/config.json" "$w" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
assert cfg.get("use_h0_action_aux") is True, cfg.get("use_h0_action_aux")
assert cfg.get("h0_action_weight") == float(sys.argv[2]), cfg.get("h0_action_weight")
print(f"W{sys.argv[2]}: checkpoint config h0 flags ok")
PY
done

[[ "$failed" -eq 0 ]] || exit 1
printf 'exit=0\n' > "$OUT/preflight.done"
