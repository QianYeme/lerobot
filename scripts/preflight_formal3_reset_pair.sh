#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/bin:$PATH"

RESETPAIR="$PROJECT/数据集/formal3/kind_merged_nomaster_reset_pair_fit48"
CAMPAIGN="$PROJECT/outputs/formal3_reset_pair_p3_20260928"
OUT_ROOT="$CAMPAIGN/preflight"
LOG_ROOT="$CAMPAIGN/logs"

[[ ! -e "$OUT_ROOT" ]] || { echo "refusing to overwrite $OUT_ROOT" >&2; exit 1; }
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

python - "$RESETPAIR" "$PROJECT/outputs/formal3_time_trim_20260928/fit48_split.json" <<'PY'
import json, sys
from pathlib import Path
import pyarrow.parquet as pq

resetpair, frozen_split = map(Path, sys.argv[1:])
new = json.loads((resetpair / "meta/formal3_fit48_manifest.json").read_text())
frozen = json.loads(frozen_split.read_text())
assert new["train"] == sorted(frozen["train"]) and new["development"] == sorted(frozen["development"])
assert len(new["train"]) == 48 and len(new["development"]) == 12
assert new["train_frames"] == 18482 and new["development_frames"] == 4695
data = pq.read_table(resetpair / "data/chunk-000/file-000.parquet", columns=["index"])
labels = pq.read_table(resetpair / "meta/reviewed_phase_labels.parquet", columns=["frame_index"])
assert len(data) == len(labels) == 23177
assert (resetpair / "annotations/top/episode_000.xml").is_file()
assert (resetpair / "meta/reset_pair_manifest.json").is_file()
PY

TRAIN_EPISODES="$(python - "$RESETPAIR" <<'PY'
import json, sys
from pathlib import Path
print(json.dumps(json.loads((Path(sys.argv[1]) / "meta/formal3_fit48_manifest.json").read_text())["train"]))
PY
)"

run_det() {
  local label="$1" data_root="$2" annotation_root="$3" repo_id="$4"
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
    --seed=1000 --batch_size=8 --num_workers=4 --steps=2 --log_freq=1 \
    --save_freq=2 --eval_freq=0 --wandb.enable=false --output_dir="$OUT_ROOT/$label"
}

run_det DET_RESETPAIR "$RESETPAIR" "$PROJECT/数据集/formal3/kind_merged_nomaster_reset_pair" \
  QYyyyyyyy/formal3_kind_merged_nomaster_reset_pair_fit48 > "$LOG_ROOT/preflight_DET_RESETPAIR.log" 2>&1

if grep -Eqi '(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)|out of memory|traceback' \
    "$LOG_ROOT"/preflight_DET_RESETPAIR.log; then
  echo "preflight log contains a failure marker" >&2
  exit 1
fi

python - "$CAMPAIGN/preflight_ready.json" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "status": "passed", "models": ["DET_RESETPAIR"],
    "concurrent_steps": 1, "batch_size_each": 8, "seed": 1000,
    "completed_at": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
PY
