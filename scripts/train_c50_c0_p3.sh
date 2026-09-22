#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/bin:$PATH"

DATA_ROOT="$PROJECT/数据集/formal1_C50_nomaster_fit32"
DATASET_REPO_ID="QYyyyyyyy/formal1_C50_nomaster_fit32"
READY="outputs/formal1_C50_nomaster_phase1/preparation/fit32_preparation_ready.json"
OUT="outputs/train/C50_NOMASTER_C0_short_s1000"
RECORDS="outputs/c50_c0_explicit_box_20260921/p3_short"

[[ -f "$READY" ]] || { echo "fit32 preparation Gate is missing" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 1; }
[[ ! -e "$RECORDS/train.log" ]] || { echo "refusing to overwrite existing P3 log" >&2; exit 1; }
mkdir -p "$RECORDS"

export DATA_ROOT
TRAIN_EPISODES="$(python - <<'PY'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ['DATA_ROOT'])
ready = json.loads(Path('outputs/formal1_C50_nomaster_phase1/preparation/fit32_preparation_ready.json').read_text())
manifest_path = root / 'meta/nomaster_manifest_fit32.json'
stats_path = root / 'meta/stats.json'
assert ready['status'] == 'passed'
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == ready['manifest_sha256']
assert hashlib.sha256(stats_path.read_bytes()).hexdigest() == ready['stats_sha256']
manifest = json.loads(manifest_path.read_text())
info = json.loads((root / 'meta/info.json').read_text())
assert info['features']['observation.state']['shape'] == [8]
assert 'master_gripper.pos' not in info['features']['observation.state']['names']
assert manifest['validation'] == [7, 11, 13, 14, 16, 27, 35, 38, 40, 43]
print(json.dumps(manifest['train']))
PY
)"

git rev-parse HEAD > "$RECORDS/code_commit.txt"
git diff HEAD -- src scripts > "$RECORDS/code_changes.patch"
cp scripts/train_c50_c0_p3.sh scripts/eval_c0_position_sensitivity.py \
  scripts/smoke_c0_explicit_box.py "$RECORDS/"
cp "$DATA_ROOT/meta/nomaster_manifest_fit32.json" "$RECORDS/dataset_manifest.json"
python -m pip freeze > "$RECORDS/environment.txt"
date -Is > "$RECORDS/started_at.txt"

cmd=(
  python -u -m lerobot.scripts.lerobot_train
  --policy.type=act_det --policy.device=cuda --policy.push_to_hub=false
  --policy.use_amp=false
  --dataset.repo_id="$DATASET_REPO_ID" --dataset.root="$DATA_ROOT"
  --dataset.episodes="$TRAIN_EPISODES" --dataset.video_backend=pyav
  --dataset.use_imagenet_stats=true --dataset.image_transforms.enable=false
  --policy.chunk_size=100 --policy.n_action_steps=1
  --policy.gripper_loss_weight=3.0 --policy.optimizer_lr_backbone=0.0001
  --policy.use_detection=true --policy.use_mask_guidance=false
  --policy.fcos_feature_inject=false --policy.mask_feature_inject=false
  --policy.use_explicit_box_condition=true
  --policy.box_condition_camera=observation.images.top
  --policy.box_condition_score_threshold=0.25
  --policy.box_condition_dropout=0.0 --policy.box_condition_noise_std=0.0
  --policy.aug_enable=false --policy.det_weight=1.0
  --policy.annotation_dir="$DATA_ROOT/annotations"
  '--policy.det_cameras={"observation.images.top":{"enable":true},"observation.images.gripper":{"enable":false}}'
  --seed=1000 --batch_size=8 --num_workers=4
  --steps=10000 --log_freq=100 --save_freq=1000
  --eval_freq=0 --wandb.enable=false --output_dir="$OUT"
)
printf '%q ' "${cmd[@]}" > "$RECORDS/command.sh"
printf '\n' >> "$RECORDS/command.sh"
"${cmd[@]}" 2>&1 | tee "$RECORDS/train.log"

python - <<'PY'
import json
from pathlib import Path
from safetensors.torch import load_file
root = Path('outputs/train/C50_NOMASTER_C0_short_s1000/checkpoints')
checkpoints = sorted(path for path in root.iterdir() if path.name.isdigit())
assert len(checkpoints) >= 2, checkpoints
first = load_file(str(checkpoints[0] / 'pretrained_model/model.safetensors'))
last = load_file(str(checkpoints[-1] / 'pretrained_model/model.safetensors'))
keys = sorted(key for key in last if 'box_condition_input_proj' in key)
deltas = {key: float((last[key] - first[key]).abs().max()) for key in keys}
result = {'first': checkpoints[0].name, 'last': checkpoints[-1].name, 'max_abs_delta': deltas,
          'passed': bool(keys) and max(deltas.values()) > 0}
Path('outputs/c50_c0_explicit_box_20260921/p3_short/box_mlp_update.json').write_text(
    json.dumps(result, indent=2), encoding='utf-8')
assert result['passed'], result
PY

python scripts/eval_c0_position_sensitivity.py \
  --checkpoint "$OUT/checkpoints/last/pretrained_model" \
  --dataset.repo_id QYyyyyyyy/formal1_C50_nomaster \
  --dataset.root "$PROJECT/数据集/formal1_C50_nomaster" \
  --episodes 33,10,6 --fixed-state-index 1 \
  --min-pan-span 0.05 --min-box-effect 0.01 --device cuda \
  --output "$RECORDS/position_sensitivity.json" \
  2>&1 | tee "$RECORDS/position_sensitivity.log"

printf 'exit=0\n' > "$RECORDS/train.done"
date -Is > "$RECORDS/finished_at.txt"
