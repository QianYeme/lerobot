#!/usr/bin/env bash
# Separate runs; start the three train-mode invocations in separate screen sessions.
set -euo pipefail
MODEL="${1:?Usage: train_c50_nomaster.sh act|det|inject smoke|train}"
MODE="${2:-train}"
case "$MODEL" in act|det|inject) ;; *) exit 2 ;; esac
case "$MODE" in smoke) STEPS=20; SAVE_FREQ=20; LOG_FREQ=1 ;; train) STEPS=100000; SAVE_FREQ=2000; LOG_FREQ=100 ;; *) exit 2 ;; esac
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/bin:$PATH"
export DATA_ROOT="$PROJECT/数据集/formal1_C50_nomaster"
PREPARATION="$PROJECT/outputs/formal1_C50_nomaster_phase1/preparation"
export TRAIN_EPISODES
TRAIN_EPISODES="$(python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['DATA_ROOT'])
manifest = json.loads((root/'meta/nomaster_manifest.json').read_text())
info = json.loads((root/'meta/info.json').read_text())
stats = json.loads((root/'meta/stats.json').read_text())
assert info['total_episodes'] == 50 and info['features']['observation.state']['shape'] == [8]
assert 'master_gripper.pos' not in info['features']['observation.state']['names']
assert stats['observation.state']['count'] == stats['action']['count'] == [28720]
assert manifest['train'] == [ep for ep in range(50) if ep not in manifest['validation']]
assert len(manifest['train']) == 40 and manifest['validation'] == [7,11,13,14,16,27,35,38,40,43]
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
  NAME=C50_NOMASTER_DET_s1000
  INJECT=false
  if [[ "$MODEL" == inject ]]; then INJECT=true; NAME=C50_NOMASTER_INJECT_s1000; fi
  extra=(
    --policy.use_detection=true --policy.use_mask_guidance=false
    --policy.fcos_feature_inject="$INJECT" '--policy.fcos_inject_levels=["p4"]'
    --policy.mask_feature_inject=false --policy.aug_enable=false --policy.det_weight=1.0
    --policy.annotation_dir="$DATA_ROOT/annotations"
    '--policy.det_cameras={"observation.images.top":{"enable":true},"observation.images.gripper":{"enable":false}}'
  )
fi
if [[ "$MODE" == smoke ]]; then NAME="${NAME%_s1000}_smoke_s1000"; fi
OUT="outputs/train/$NAME"
RECORDS="outputs/formal1_C50_nomaster_phase1/$NAME"
[[ ! -e "$OUT" && ! -e "$RECORDS" ]] || { echo "Existing run $NAME; refusing overwrite." >&2; exit 1; }
mkdir -p "$RECORDS"
git rev-parse HEAD > "$RECORDS/code_commit.txt"
git diff HEAD -- src scripts > "$RECORDS/code_changes.patch"
cp scripts/train_c50_nomaster.sh "$RECORDS/train_script.sh"
cp "$DATA_ROOT/meta/nomaster_manifest.json" "$RECORDS/dataset_manifest.json"
python -m pip freeze > "$RECORDS/environment.txt"
date -Is > "$RECORDS/started_at.txt"
cmd=(
  python -u -m lerobot.scripts.lerobot_train
  --policy.type="$POLICY_TYPE" --policy.device=cuda --policy.push_to_hub=false
  --policy.use_amp=false
  --dataset.repo_id=QYyyyyyyy/formal1_C50_nomaster --dataset.root="$DATA_ROOT"
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
printf 'exit=0\n' > "$RECORDS/train.done"
date -Is > "$RECORDS/finished_at.txt"
