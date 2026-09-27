#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
[[ "$MODE" == preflight || "$MODE" == train ]] || {
  echo 'usage: train_formal3_mixed_4way.sh [preflight|train]' >&2
  exit 2
}

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/bin:$PATH"

DATA_ROOT="$PROJECT/数据集/formal3/kind_merged_nomaster_fit48"
SOURCE_ROOT="$PROJECT/数据集/formal3/kind_merged_nomaster"
PHASE_LABELS="$PROJECT/outputs/formal3_phase_labels_20260927/final_reviewed_labels/reviewed_phase_labels.parquet"
CAMPAIGN="$PROJECT/outputs/formal3_mixed_p4_20260927"
SPLIT_READY="$PROJECT/outputs/formal3_phase_p3_20260927/split_ready.json"
READY="$CAMPAIGN/preparation_ready.json"

python - "$SPLIT_READY" "$DATA_ROOT" "$SOURCE_ROOT" "$PHASE_LABELS" <<'PY'
import json, sys
from pathlib import Path

split = json.loads(Path(sys.argv[1]).read_text())
root, source, labels = map(Path, sys.argv[2:])
manifest = json.loads((root / "meta/formal3_fit48_manifest.json").read_text())
assert split["status"] == "passed"
assert split["train"] == manifest["train"] and split["development"] == manifest["development"]
assert len(manifest["train"]) == 48 and len(manifest["development"]) == 12
assert labels.is_file() and (source / "annotations/top/episode_000.xml").is_file()
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
  [[ -f "$READY" ]] || { echo "missing preparation Gate: $READY" >&2; exit 1; }
  grep -q '状态：`AUTHORIZED`' "$CAMPAIGN/experiment.md" || {
    echo 'experiment is not authorized' >&2
    exit 1
  }
  STEPS=100000
  SAVE_FREQ=20000
  OUT_ROOT="$PROJECT/outputs/train"
fi
mkdir -p "$CAMPAIGN/logs" "$OUT_ROOT"

run_act_det() {
  local label="$1" box_mode="$2" use_phase="$3" phase_weight="$4"
  local output="$OUT_ROOT/$label"
  [[ "$MODE" == preflight ]] || output="$OUT_ROOT/F3_P4_${label}_s1000"
  [[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; return 1; }
  python -u -m lerobot.scripts.lerobot_train \
    --policy.type=act_det --policy.device=cuda --policy.push_to_hub=false --policy.use_amp=false \
    --dataset.repo_id=QYyyyyyyy/formal3_kind_merged_nomaster_fit48 --dataset.root="$DATA_ROOT" \
    --dataset.episodes="$TRAIN_EPISODES" --dataset.video_backend=pyav \
    --dataset.use_imagenet_stats=true --dataset.image_transforms.enable=false \
    --policy.chunk_size=100 --policy.n_action_steps=1 --policy.gripper_loss_weight=3.0 \
    --policy.optimizer_lr_backbone=0.0001 --policy.use_detection=true --policy.use_mask_guidance=false \
    --policy.fcos_feature_inject=false --policy.mask_feature_inject=false \
    --policy.use_explicit_box_condition="$box_mode" --policy.box_condition_mode=state \
    --policy.box_condition_camera=observation.images.top --policy.box_condition_score_threshold=0.25 \
    --policy.box_condition_dropout=0.0 --policy.box_condition_noise_std=0.0 \
    --policy.use_phase_aux="$use_phase" --policy.phase_labels="$PHASE_LABELS" \
    --policy.phase_weight="$phase_weight" --policy.phase_num_classes=4 \
    --policy.aug_enable=false --policy.det_weight=1.0 --policy.annotation_dir="$SOURCE_ROOT/annotations" \
    '--policy.det_cameras={"observation.images.top":{"enable":true},"observation.images.gripper":{"enable":false}}' \
    --seed=1000 --batch_size=8 --num_workers=4 --steps="$STEPS" --log_freq=1 \
    --save_freq="$SAVE_FREQ" --eval_freq=0 --wandb.enable=false --output_dir="$output"
}

run_diffusion() {
  local output="$OUT_ROOT/DIFFUSION_BASE"
  [[ "$MODE" == preflight ]] || output="$OUT_ROOT/F3_P4_DIFFUSION_BASE_s1000"
  [[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; return 1; }
  python -u -m lerobot.scripts.lerobot_train \
    --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false --policy.use_amp=false \
    --dataset.repo_id=QYyyyyyyy/formal3_kind_merged_nomaster_fit48 --dataset.root="$DATA_ROOT" \
    --dataset.episodes="$TRAIN_EPISODES" --dataset.video_backend=pyav \
    --dataset.use_imagenet_stats=true --dataset.image_transforms.enable=false \
    --policy.horizon=16 --policy.n_obs_steps=1 --policy.n_action_steps=1 --policy.num_inference_steps=10 \
    --policy.drop_n_last_frames=0 --policy.vision_backbone=resnet18 \
    --seed=1000 --batch_size=8 --num_workers=4 --steps="$STEPS" --log_freq=1 \
    --save_freq="$SAVE_FREQ" --eval_freq=0 --wandb.enable=false --output_dir="$output"
}

run_diffusion > "$CAMPAIGN/logs/${MODE}_DIFFUSION_BASE.log" 2>&1 & p1=$!
run_act_det DET_BASE false false 0.0 > "$CAMPAIGN/logs/${MODE}_DET_BASE.log" 2>&1 & p2=$!
run_act_det DET_C1_STATE true false 0.0 > "$CAMPAIGN/logs/${MODE}_DET_C1_STATE.log" 2>&1 & p3=$!
run_act_det DET_PHASE_W010 false true 0.10 > "$CAMPAIGN/logs/${MODE}_DET_PHASE_W010.log" 2>&1 & p4=$!

failed=0
for pair in "DIFFUSION_BASE:$p1" "DET_BASE:$p2" "DET_C1_STATE:$p3" "DET_PHASE_W010:$p4"; do
  label="${pair%%:*}"
  pid="${pair##*:}"
  if wait "$pid"; then
    echo 0 > "$CAMPAIGN/logs/${MODE}_${label}.exit"
  else
    code=$?
    echo "$code" > "$CAMPAIGN/logs/${MODE}_${label}.exit"
    failed=1
  fi
done
[[ "$failed" -eq 0 ]] || { echo 'at least one worker failed' >&2; exit 1; }

if [[ "$MODE" == preflight ]]; then
  python - "$READY" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "status": "passed", "four_way_concurrent_steps": 2, "batch_size_each": 8,
    "completed_at": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
PY
else
  printf 'exit=0\n' > "$CAMPAIGN/train.done"
  date -Is > "$CAMPAIGN/finished_at.txt"
fi
