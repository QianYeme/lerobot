#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-train}"
[[ "$MODE" == preflight || "$MODE" == train ]] || { echo 'usage: train_formal3_phase_4way.sh [preflight|train]' >&2; exit 2; }
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/bin:$PATH"

DATA_ROOT="$PROJECT/数据集/formal3/kind_merged_nomaster_fit48"
SOURCE_ROOT="$PROJECT/数据集/formal3/kind_merged_nomaster"
PHASE_LABELS="$PROJECT/outputs/formal3_phase_labels_20260927/final_reviewed_labels/reviewed_phase_labels.parquet"
CAMPAIGN="$PROJECT/outputs/formal3_phase_p3_20260927"
READY="$CAMPAIGN/preparation_ready.json"
SPLIT_READY="$CAMPAIGN/split_ready.json"
[[ -f "$SPLIT_READY" ]] || { echo "missing split Gate: $SPLIT_READY" >&2; exit 1; }
if [[ "$MODE" == train ]]; then
  [[ -f "$READY" ]] || { echo "missing preparation Gate: $READY" >&2; exit 1; }
fi
python - "$SPLIT_READY" "$DATA_ROOT" "$PHASE_LABELS" <<'PY'
import json, sys
from pathlib import Path
ready = json.loads(Path(sys.argv[1]).read_text())
assert ready['status'] == 'passed'
root, labels = Path(sys.argv[2]), Path(sys.argv[3])
manifest = json.loads((root/'meta/formal3_fit48_manifest.json').read_text())
assert len(manifest['train']) == 48 and len(manifest['development']) == 12
assert not set(manifest['train']) & set(manifest['development'])
assert labels.is_file() and (root/'annotations/top/episode_000.xml').is_file()
PY

TRAIN_EPISODES="$(python - "$DATA_ROOT" <<'PY'
import json, sys
from pathlib import Path
print(json.dumps(json.loads((Path(sys.argv[1])/'meta/formal3_fit48_manifest.json').read_text())['train']))
PY
)"
if [[ "$MODE" == preflight ]]; then
  STEPS=2; SAVE_FREQ=2; OUT_ROOT="$CAMPAIGN/preflight"
else
  STEPS=100000; SAVE_FREQ=20000; OUT_ROOT="$PROJECT/outputs/train"
  [[ -f "$CAMPAIGN/preflight.done" ]] || { echo 'preflight.done missing' >&2; exit 1; }
  grep -q '状态：`AUTHORIZED`' "$CAMPAIGN/experiment.md" || { echo 'experiment is not authorized' >&2; exit 1; }
fi
mkdir -p "$CAMPAIGN/logs" "$OUT_ROOT"

run_one() {
  local label="$1" use_phase="$2" phase_weight="$3"
  local output
  if [[ "$MODE" == preflight ]]; then output="$OUT_ROOT/$label"; else output="$OUT_ROOT/F3_P3_${label}_s1000"; fi
  [[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; return 1; }
  python -u -m lerobot.scripts.lerobot_train \
    --policy.type=act_det --policy.device=cuda --policy.push_to_hub=false --policy.use_amp=false \
    --dataset.repo_id=formal3/kind_merged_nomaster_fit48 --dataset.root="$DATA_ROOT" \
    --dataset.episodes="$TRAIN_EPISODES" --dataset.video_backend=pyav \
    --dataset.use_imagenet_stats=true --dataset.image_transforms.enable=false \
    --policy.chunk_size=100 --policy.n_action_steps=1 --policy.gripper_loss_weight=3.0 \
    --policy.optimizer_lr_backbone=0.0001 --policy.use_detection=true --policy.use_mask_guidance=false \
    --policy.fcos_feature_inject=false --policy.mask_feature_inject=false \
    --policy.use_explicit_box_condition=false --policy.aug_enable=false --policy.det_weight=1.0 \
    --policy.annotation_dir="$SOURCE_ROOT/annotations" \
    '--policy.det_cameras={"observation.images.top":{"enable":true},"observation.images.gripper":{"enable":false}}' \
    --policy.use_phase_aux="$use_phase" --policy.phase_labels="$PHASE_LABELS" \
    --policy.phase_weight="$phase_weight" --policy.phase_num_classes=4 \
    --seed=1000 --batch_size=8 --num_workers=4 --steps="$STEPS" --log_freq=1 \
    --save_freq="$SAVE_FREQ" --eval_freq=0 --wandb.enable=false --output_dir="$output"
}

run_one DET_BASE false 0.0 > "$CAMPAIGN/logs/${MODE}_DET_BASE.log" 2>&1 & p1=$!
run_one PHASE_W003 true 0.03 > "$CAMPAIGN/logs/${MODE}_PHASE_W003.log" 2>&1 & p2=$!
run_one PHASE_W010 true 0.10 > "$CAMPAIGN/logs/${MODE}_PHASE_W010.log" 2>&1 & p3=$!
run_one PHASE_W030 true 0.30 > "$CAMPAIGN/logs/${MODE}_PHASE_W030.log" 2>&1 & p4=$!

failed=0
for pair in "DET_BASE:$p1" "PHASE_W003:$p2" "PHASE_W010:$p3" "PHASE_W030:$p4"; do
  label="${pair%%:*}"; pid="${pair##*:}"
  if wait "$pid"; then echo 0 > "$CAMPAIGN/logs/${MODE}_${label}.exit"; else code=$?; echo "$code" > "$CAMPAIGN/logs/${MODE}_${label}.exit"; failed=1; fi
done
[[ "$failed" -eq 0 ]] || { echo 'at least one worker failed' >&2; exit 1; }
if [[ "$MODE" == preflight ]]; then
  printf 'exit=0\n' > "$CAMPAIGN/preflight.done"
  python - "$CAMPAIGN/preparation_ready.json" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    'status': 'passed', 'four_way_concurrent_steps': 2, 'batch_size_each': 8,
    'completed_at': datetime.now(timezone.utc).isoformat(),
}, indent=2) + '\n')
PY
else
  printf 'exit=0\n' > "$CAMPAIGN/train.done"
  date -Is > "$CAMPAIGN/finished_at.txt"
fi
