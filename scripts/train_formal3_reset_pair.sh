#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/bin:$PATH"

RESETPAIR="$PROJECT/数据集/formal3/kind_merged_nomaster_reset_pair_fit48"
CAMPAIGN="$PROJECT/outputs/formal3_reset_pair_p3_20260928"
OUT_ROOT="$PROJECT/outputs/train"
LOG_ROOT="$CAMPAIGN/logs"
READY="$CAMPAIGN/preflight_ready.json"
CONTRACT="$CAMPAIGN/experiment.md"

python - "$READY" <<'PY'
import json, sys
from pathlib import Path
ready = json.loads(Path(sys.argv[1]).read_text())
assert ready["status"] == "passed" and ready["concurrent_steps"] == 1
assert ready["models"] == ["DET_RESETPAIR"]
PY
grep -q '状态：`AUTHORIZED`' "$CONTRACT" || {
  echo "experiment contract is not AUTHORIZED" >&2
  exit 1
}

TRAIN_EPISODES="$(python - "$RESETPAIR" <<'PY'
import json, sys
from pathlib import Path
print(json.dumps(json.loads((Path(sys.argv[1]) / "meta/formal3_fit48_manifest.json").read_text())["train"]))
PY
)"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

run_det() {
  local label="$1" data_root="$2" annotation_root="$3" repo_id="$4"
  local output="$OUT_ROOT/$label"
  [[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; return 1; }
  python -u -m lerobot.scripts.lerobot_train \
    --policy.type=act_det --policy.device=cuda --policy.push_to_hub=false --policy.use_amp=false \
    --dataset.repo_id="$repo_id" --dataset.root="$data_root" \
    --dataset.episodes="$TRAIN_EPISODES" --dataset.video_backend=pyav \
    --dataset.use_imagenet_stats=true --dataset.image_transforms.enable=false \
    --policy.chunk_size=100 --policy.n_action_steps=1 --policy.gripper_loss_weight=3.0 \
    --policy.optimizer_lr_backbone=0.0001 --policy.use_detection=true \
    --policy.use_mask_guidance=false --policy.fcos_feature_inject=false \
    --policy.mask_feature_inject=false --policy.use_explicit_box_condition=false \
    --policy.use_phase_aux=false --policy.aug_enable=false \
    --policy.det_weight=1.0 --policy.annotation_dir="$annotation_root/annotations" \
    '--policy.det_cameras={"observation.images.top":{"enable":true},"observation.images.gripper":{"enable":false}}' \
    --seed=1000 --batch_size=8 --num_workers=4 --steps=10000 --log_freq=100 \
    --save_freq=2000 --eval_freq=0 --wandb.enable=false --output_dir="$output"
}

run_det F3_P3_DET_RESETPAIR_10K_s1000 "$RESETPAIR" \
  "$PROJECT/数据集/formal3/kind_merged_nomaster_reset_pair" QYyyyyyyy/formal3_kind_merged_nomaster_reset_pair_fit48 \
  > "$LOG_ROOT/train_DET_RESETPAIR_10K.log" 2>&1 &
p1=$!

failed=0
if wait "$p1"; then code=0; else code=$?; failed=1; fi
echo "$code" > "$LOG_ROOT/train_DET_RESETPAIR_10K.exit"
[[ "$failed" -eq 0 ]] || { echo "10k worker failed" >&2; exit 1; }

if grep -Eqi '(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)|out of memory|traceback' \
    "$LOG_ROOT"/train_DET_RESETPAIR_10K.log; then
  echo "10k log contains a failure marker" >&2
  exit 1
fi

printf 'exit=0\n' > "$CAMPAIGN/train.done"
date -Is > "$CAMPAIGN/finished_at.txt"
