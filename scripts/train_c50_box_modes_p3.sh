#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/bin:$PATH"

DATA_ROOT="$PROJECT/数据集/formal1_C50_nomaster_fit32"
DATASET_REPO_ID="QYyyyyyyy/formal1_C50_nomaster_fit32"
RECORDS="outputs/c50_box_condition_modes_20260921/p3_short"
[[ -f "$RECORDS/preflight.done" ]] || { echo "preflight Gate is missing" >&2; exit 1; }
grep -q '状态：`READY`' "$RECORDS/experiment.md" || { echo "experiment contract is not READY" >&2; exit 1; }
[[ ! -e "$RECORDS/train.done" ]] || { echo "training already completed" >&2; exit 1; }
mkdir -p "$RECORDS/logs" "$RECORDS/eval"

export DATA_ROOT
TRAIN_EPISODES="$(python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['DATA_ROOT'])
manifest = json.loads((root / 'meta/nomaster_manifest_fit32.json').read_text())
print(json.dumps(manifest['train']))
PY
)"

git rev-parse HEAD > "$RECORDS/code_commit.txt"
git diff HEAD -- src scripts tests > "$RECORDS/code_changes.patch"
python -m pip freeze > "$RECORDS/environment.txt"
date -Is > "$RECORDS/started_at.txt"

run_mode() {
  local label="$1"
  local mode="$2"
  local out="outputs/train/C50_NOMASTER_${label}_fair_s1000"
  [[ ! -e "$out" ]] || { echo "refusing to overwrite $out" >&2; return 1; }
  python -u -m lerobot.scripts.lerobot_train \
    --policy.type=act_det --policy.device=cuda --policy.push_to_hub=false --policy.use_amp=false \
    --dataset.repo_id="$DATASET_REPO_ID" --dataset.root="$DATA_ROOT" \
    --dataset.episodes="$TRAIN_EPISODES" --dataset.video_backend=pyav \
    --dataset.use_imagenet_stats=true --dataset.image_transforms.enable=false \
    --policy.chunk_size=100 --policy.n_action_steps=1 \
    --policy.gripper_loss_weight=3.0 --policy.optimizer_lr_backbone=0.0001 \
    --policy.use_detection=true --policy.use_mask_guidance=false \
    --policy.fcos_feature_inject=false --policy.mask_feature_inject=false \
    --policy.use_explicit_box_condition=true --policy.box_condition_mode="$mode" \
    --policy.box_condition_camera=observation.images.top \
    --policy.box_condition_score_threshold=0.25 \
    --policy.box_condition_dropout=0.0 --policy.box_condition_noise_std=0.0 \
    --policy.box_action_residual_alpha=0.05 \
    --policy.aug_enable=false --policy.det_weight=1.0 \
    --policy.annotation_dir="$DATA_ROOT/annotations" \
    '--policy.det_cameras={"observation.images.top":{"enable":true},"observation.images.gripper":{"enable":false}}' \
    --seed=1000 --batch_size=8 --num_workers=4 \
    --steps=10000 --log_freq=100 --save_freq=1000 --eval_freq=0 --wandb.enable=false \
    --output_dir="$out"
}

labels=(C0 C1 C2)
modes=(token state action_residual)
pids=()
for index in "${!labels[@]}"; do
  run_mode "${labels[$index]}" "${modes[$index]}" \
    > "$RECORDS/logs/${labels[$index]}.log" 2>&1 &
  pids+=("$!")
  echo "${pids[-1]}" > "$RECORDS/logs/${labels[$index]}.pid"
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo 0 > "$RECORDS/logs/${labels[$index]}.exit"
  else
    status=$?
    echo "$status" > "$RECORDS/logs/${labels[$index]}.exit"
    failed=1
  fi
done
[[ "$failed" -eq 0 ]] || { echo "at least one 10k training run failed" >&2; exit 1; }

for label in "${labels[@]}"; do
  for step in 004000 006000 008000 010000; do
    python scripts/eval_box_condition_modes.py \
      --checkpoint "outputs/train/C50_NOMASTER_${label}_fair_s1000/checkpoints/$step/pretrained_model" \
      --dataset.repo_id QYyyyyyyy/formal1_C50_nomaster \
      --dataset.root "$PROJECT/数据集/formal1_C50_nomaster" \
      --episodes 33,10,6 --fixed-state-index 1 --device cuda \
      --min-pan-span 0.05 --min-box-effect 0.01 \
      --output "$RECORDS/eval/${label}_${step}.json" \
      > "$RECORDS/eval/${label}_${step}.log" 2>&1 || true
  done
done

python - <<'PY'
import json
from pathlib import Path
root = Path('outputs/c50_box_condition_modes_20260921/p3_short/eval')
result = {'status': 'passed', 'modes': {}}
for label in ('C0', 'C1', 'C2'):
    rows = [json.loads((root / f'{label}_{step}.json').read_text())
            for step in ('004000', '006000', '008000', '010000')]
    final = rows[-1]
    stable_direction = all(row['gates']['coordinate_only_direction_matches_gt'] for row in rows)
    mode_pass = final['status'] == 'passed' and stable_direction
    if label == 'C2':
        alpha = final['residual']['alpha']
        mode_pass = mode_pass and 0.01 <= alpha <= 0.95
    result['modes'][label] = {
        'passed': mode_pass,
        'stable_direction_4k_to_10k': stable_direction,
        'final': final['gates'],
        'residual': final['residual'],
    }
result['status'] = 'passed' if any(row['passed'] for row in result['modes'].values()) else 'failed'
Path('outputs/c50_box_condition_modes_20260921/p3_short/validation.json').write_text(
    json.dumps(result, indent=2), encoding='utf-8')
if result['status'] != 'passed':
    raise SystemExit(1)
PY

printf 'exit=0\n' > "$RECORDS/train.done"
date -Is > "$RECORDS/finished_at.txt"
