#!/usr/bin/env bash
# Separate runs; start the train-mode invocations in separate screen sessions.
# Models: act | det | i0 (FCOS token inject, legacy alias inject) | i1 (p4 residual inject).
# Modes:  smoke (20 steps, train40 stats) | short (10k steps, fit32 stats; phase C) | train (100k, train40; phase E).
set -euo pipefail
MODEL="${1:?Usage: train_c50_nomaster.sh act|det|i0|i1 smoke|short|train}"
MODE="${2:-train}"
case "$MODEL" in act|det|i0|i1|inject) ;; *) exit 2 ;; esac
case "$MODE" in
  smoke) STEPS=20; SAVE_FREQ=20; LOG_FREQ=1 ;;
  short) STEPS=10000; SAVE_FREQ=1000; LOG_FREQ=100 ;;
  train) STEPS=100000; SAVE_FREQ=10000; LOG_FREQ=100 ;;
  *) exit 2 ;;
esac
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/bin:$PATH"
export TRAIN_SPLIT_MODE="$MODE"
FIT_SPLIT="outputs/mask_inject_phase3_20260918/dev_split_with_stats.json"
if [[ "$MODE" == short ]]; then
  export DATA_ROOT="$PROJECT/数据集/formal1_C50_nomaster_fit32"
  PREPARATION="$PROJECT/outputs/formal1_C50_nomaster_phase1/preparation"
  [[ -f "$PREPARATION/fit32_preparation_ready.json" ]] || { echo 'fit32 preparation has not passed; run scripts/prepare_c50_fit32_dataset.py first.' >&2; exit 1; }
  python - <<'PY'
import hashlib, json, os
from pathlib import Path
ready = json.loads(Path('outputs/formal1_C50_nomaster_phase1/preparation/fit32_preparation_ready.json').read_text())
assert ready['status'] == 'passed'
for key, path in [('stats_sha256', Path(os.environ['DATA_ROOT'])/'meta/stats.json'),
                  ('manifest_sha256', Path(os.environ['DATA_ROOT'])/'meta/nomaster_manifest_fit32.json')]:
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ready[key], f'Changed preparation artifact: {path}'
PY
  DATASET_REPO_ID=QYyyyyyyy/formal1_C50_nomaster_fit32
else
  export DATA_ROOT="$PROJECT/数据集/formal1_C50_nomaster"
  PREPARATION="$PROJECT/outputs/formal1_C50_nomaster_phase1/preparation"
  DATASET_REPO_ID=QYyyyyyyy/formal1_C50_nomaster
fi
export TRAIN_EPISODES
TRAIN_EPISODES="$(python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['DATA_ROOT'])
info = json.loads((root/'meta/info.json').read_text())
stats = json.loads((root/'meta/stats.json').read_text())
assert info['total_episodes'] == 50 and info['features']['observation.state']['shape'] == [8]
assert 'master_gripper.pos' not in info['features']['observation.state']['names']
if os.environ['TRAIN_SPLIT_MODE'] == 'short':
    manifest = json.loads((root/'meta/nomaster_manifest_fit32.json').read_text())
    expected_frames = 22976
else:
    manifest = json.loads((root/'meta/nomaster_manifest.json').read_text())
    expected_frames = 28720
assert stats['observation.state']['count'] == stats['action']['count'] == [expected_frames]
assert manifest['validation'] == [7, 11, 13, 14, 16, 27, 35, 38, 40, 43]
print(json.dumps(manifest['train']))
PY
)"
if [[ "$MODE" == train ]]; then
  [[ -f "$PREPARATION/preparation_ready.json" ]] || { echo 'Preparation has not passed; refusing full training.' >&2; exit 1; }
  python - <<'PY'
import hashlib, json, os
from pathlib import Path
ready = json.loads(Path('outputs/formal1_C50_nomaster_phase1/preparation/preparation_ready.json').read_text())
assert ready['status'] == 'passed'
for key, path in [('training_script_sha256', Path('scripts/train_c50_nomaster.sh')),
                  ('manifest_sha256', Path(os.environ['DATA_ROOT'])/'meta/nomaster_manifest.json'),
                  ('statistics_sha256', Path(os.environ['DATA_ROOT'])/'meta/stats.json')]:
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ready[key], f'Changed preparation artifact: {path}'
PY
fi
POLICY_TYPE=act
NAME=C50_NOMASTER_ACT_s1000
extra=()
if [[ "$MODEL" != act ]]; then
  POLICY_TYPE=act_det
  INJECT=false
  INJECT_MODE=tokens
  case "$MODEL" in
    det) NAME=C50_NOMASTER_DET_s1000 ;;
    i0|inject) NAME=C50_NOMASTER_I0_s1000; INJECT=true ;;
    i1) NAME=C50_NOMASTER_I1_s1000; INJECT=true; INJECT_MODE=residual ;;
  esac
  extra=(
    --policy.use_detection=true --policy.use_mask_guidance=false
    --policy.fcos_feature_inject="$INJECT" --policy.fcos_inject_mode="$INJECT_MODE"
    '--policy.fcos_inject_levels=["p4"]'
    --policy.mask_feature_inject=false --policy.aug_enable=false --policy.det_weight=1.0
    --policy.annotation_dir="$DATA_ROOT/annotations"
    '--policy.det_cameras={"observation.images.top":{"enable":true},"observation.images.gripper":{"enable":false}}'
  )
fi
if [[ "$MODE" == smoke ]]; then NAME="${NAME%_s1000}_smoke_s1000"; fi
if [[ "$MODE" == short ]]; then NAME="${NAME%_s1000}_short_s1000"; fi
export NAME
OUT="outputs/train/$NAME"
RECORDS="outputs/formal1_C50_nomaster_phase1/$NAME"
[[ ! -e "$OUT" && ! -e "$RECORDS" ]] || { echo "Existing run $NAME; refusing overwrite." >&2; exit 1; }
mkdir -p "$RECORDS"
git rev-parse HEAD > "$RECORDS/code_commit.txt"
git diff HEAD -- src scripts > "$RECORDS/code_changes.patch"
cp scripts/train_c50_nomaster.sh "$RECORDS/train_script.sh"
cp "$DATA_ROOT/meta/nomaster_manifest_fit32.json" "$RECORDS/dataset_manifest.json" 2>/dev/null \
  || cp "$DATA_ROOT/meta/nomaster_manifest.json" "$RECORDS/dataset_manifest.json"
python -m pip freeze > "$RECORDS/environment.txt"
date -Is > "$RECORDS/started_at.txt"
cmd=(
  python -u -m lerobot.scripts.lerobot_train
  --policy.type="$POLICY_TYPE" --policy.device=cuda --policy.push_to_hub=false
  --policy.use_amp=false
  --dataset.repo_id="$DATASET_REPO_ID" --dataset.root="$DATA_ROOT"
  --dataset.episodes="$TRAIN_EPISODES" --dataset.video_backend=pyav
  --dataset.use_imagenet_stats=true --dataset.image_transforms.enable=false
  --policy.chunk_size=100 --policy.n_action_steps=1
  --policy.gripper_loss_weight=3.0 --policy.optimizer_lr_backbone=0.0001
  --seed=1000 --batch_size=8 --num_workers=4
  --steps="$STEPS" --log_freq="$LOG_FREQ" --save_freq="$SAVE_FREQ"
  --eval_freq=0 --wandb.enable=false --output_dir="$OUT"
  "${extra[@]}"
)
printf '%q ' "${cmd[@]}" > "$RECORDS/command.sh"
printf '\n' >> "$RECORDS/command.sh"
"${cmd[@]}" 2>&1 | tee "$RECORDS/train.log"
if [[ "$MODE" == short ]]; then
  python - <<'PY'
import json, os, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from check_c50_processor_stats import compare
name = os.environ.get('NAME')
report = compare(f"outputs/train/{name}/checkpoints/last/pretrained_model",
                 "outputs/mask_inject_phase3_20260918/dev_split_with_stats.json")
Path(f"outputs/formal1_C50_nomaster_phase1/{name}/processor_check.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
assert report['status'] == 'passed', 'Saved normalization stats differ from fit32 statistics'
PY
fi
printf 'exit=0\n' > "$RECORDS/train.done"
date -Is > "$RECORDS/finished_at.txt"
