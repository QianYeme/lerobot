#!/usr/bin/env bash
set -euo pipefail
CANDIDATE="${1:?Usage: train_n0_overnight_candidate.sh r2|r1|r3 train|smoke}"
MODE="${2:-train}"
case "$CANDIDATE" in r2|r1|r3) ;; *) exit 2 ;; esac
case "$MODE" in smoke) STEPS=2; SAVE_FREQ=2; LOG_FREQ=1 ;; train) STEPS=100000; SAVE_FREQ=10000; LOG_FREQ=100 ;; *) exit 2 ;; esac
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONNOUSERSITE=1; export PATH="/root/miniconda3/bin:$PATH"
DATA_ROOT="$PROJECT/formal4_data/kind_merged_nomaster"; REPO_ID="formal4/kind_merged_nomaster"
PROTOTYPES="$PROJECT/experiments/position/p0_1_spatial_pooling_fix/index1/result/cup_prototype_bank_v1.pt"
TRAIN_EPISODES='[0,2,4,6,7,9,10,13,14,17,18,19,22,23,24,25,26,29,30,32,34,35,36,38,40,42,43,45,48,49,50,51,54,55,56,57]'
case "$CANDIDATE" in
  r2) LABEL="R2_ACTION_LEVEL"; EXTRA=(--policy.action_level_spatial_conditioning=true) ;;
  r1) LABEL="R1_RELATIONAL"; EXTRA=(--policy.relational_state_spatial_conditioning=true) ;;
  r3) LABEL="R3_FILM"; EXTRA=(--policy.spatial_film_conditioning=true) ;;
esac
NAME="${LABEL}_s1"; [[ "$MODE" == smoke ]] && NAME="${NAME}_smoke${RUN_SUFFIX:+_$RUN_SUFFIX}"
OUT="results/experiments/N0_OVERNIGHT_SPATIAL_CONDITIONING_BATCH/${NAME}"
RECORDS="experiments/position/n0_overnight_batch/index1/result/${LABEL}"
[[ ! -e "$OUT" && ! -e "$RECORDS/runs/$NAME" ]] || { echo "Existing run $NAME; refusing overwrite." >&2; exit 1; }
mkdir -p "$RECORDS/runs/$NAME"; RECORDS="$RECORDS/runs/$NAME"
git rev-parse HEAD > "$RECORDS/code_commit.txt"; git diff HEAD -- src scripts > "$RECORDS/code_changes.patch"; cp "$0" "$RECORDS/train_script.sh"
date -Is > "$RECORDS/started_at.txt"; nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv > "$RECORDS/gpu_start.csv"; python -m pip freeze > "$RECORDS/environment.txt"
BASE=(--policy.model_type=oc_act --policy.object_conditioning_enabled=true --policy.object_conditioning_top_camera_key=observation.images.top --policy.object_conditioning_dino_model=dinov2_vits14 --policy.object_conditioning_dino_frozen=true --policy.object_conditioning_prototype_path="$PROTOTYPES" --policy.object_conditioning_num_prototypes=4 --policy.object_conditioning_num_object_tokens=5 --policy.object_conditioning_adapter_dim=256 --policy.object_conditioning_gate_init=0.0 --policy.object_conditioning_temperature=0.20 --policy.explicit_spatial_conditioning=false --policy.joint_state_spatial_conditioning=false --policy.action_level_spatial_conditioning=false --policy.relational_state_spatial_conditioning=false --policy.spatial_film_conditioning=false)
cmd=(python -u -m lerobot.scripts.lerobot_train --policy.type=act --policy.device=cuda --policy.push_to_hub=false --policy.use_amp=false --dataset.repo_id="$REPO_ID" --dataset.root="$DATA_ROOT" --dataset.episodes="$TRAIN_EPISODES" --dataset.video_backend=pyav --dataset.use_imagenet_stats=true --dataset.image_transforms.enable=false --policy.chunk_size=100 --policy.n_action_steps=1 --policy.gripper_loss_weight=3.0 --policy.optimizer_lr_backbone=0.0001 --seed=1 --batch_size=4 --num_workers=4 --steps="$STEPS" --log_freq="$LOG_FREQ" --save_freq="$SAVE_FREQ" --eval_freq=0 --wandb.enable=false --output_dir="$OUT" "${BASE[@]}" "${EXTRA[@]}")
printf '%q ' "${cmd[@]}" > "$RECORDS/command.sh"
printf '\n' >> "$RECORDS/command.sh"
"${cmd[@]}" 2>&1 | tee "$RECORDS/train.log"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv > "$RECORDS/gpu_end.csv"
printf 'exit=0\n' > "$RECORDS/train.done"
date -Is > "$RECORDS/finished_at.txt"
