#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:?Usage: train_t2_fixed_scale.sh b2|p3 train|smoke}"
MODE="${2:-train}"
case "$MODEL" in b2|p3) ;; *) exit 2 ;; esac
case "$MODE" in smoke) STEPS=2; SAVE_FREQ=2; LOG_FREQ=1 ;; train) STEPS=100000; SAVE_FREQ=10000; LOG_FREQ=100 ;; *) exit 2 ;; esac
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PATH="/root/miniconda3/bin:$PATH"
DATA_ROOT="$PROJECT/formal4_data/kind_merged_nomaster"
REPO_ID="formal4/kind_merged_nomaster"
PROTOTYPES="$PROJECT/experiments/position/p0_1_spatial_pooling_fix/index1/result/cup_prototype_bank_v1.pt"
TRAIN_EPISODES='[0,2,4,6,7,9,10,13,14,17,18,19,22,23,24,25,26,29,30,32,34,35,36,38,40,42,43,45,48,49,50,51,54,55,56,57]'
NAME="T2_${MODEL^^}_s1"; [[ "$MODE" == smoke ]] && NAME="${NAME}_smoke"
OUT="results/experiments/T2_FIXED_SCALE_SPATIAL_CONDITIONING/${NAME}"
RECORDS="experiments/position/t2_fixed_scale_spatial/index1/result/runs/${NAME}"
[[ ! -e "$OUT" && ! -e "$RECORDS" ]] || { echo "Existing run $NAME; refusing overwrite." >&2; exit 1; }
mkdir -p "$RECORDS"
git rev-parse HEAD > "$RECORDS/code_commit.txt"
git diff HEAD -- src scripts > "$RECORDS/code_changes.patch"
cp "$0" "$RECORDS/train_script.sh"
date -Is > "$RECORDS/started_at.txt"
python -m pip freeze > "$RECORDS/environment.txt"
extra=(--policy.model_type=oc_act --policy.object_conditioning_enabled=true --policy.object_conditioning_top_camera_key=observation.images.top --policy.object_conditioning_dino_model=dinov2_vits14 --policy.object_conditioning_dino_frozen=true --policy.object_conditioning_prototype_path="$PROTOTYPES" --policy.object_conditioning_num_prototypes=4 --policy.object_conditioning_num_object_tokens=5 --policy.object_conditioning_adapter_dim=256 --policy.object_conditioning_gate_init=0.0 --policy.object_conditioning_temperature=0.20 --policy.explicit_spatial_conditioning=false)
if [[ "$MODEL" == p3 ]]; then extra+=(--policy.explicit_spatial_conditioning=true --policy.explicit_spatial_dim=128 --policy.explicit_spatial_beta_init=0.01 --policy.explicit_spatial_fixed_scale=0.5); fi
cmd=(python -u -m lerobot.scripts.lerobot_train --policy.type=act --policy.device=cuda --policy.push_to_hub=false --policy.use_amp=false --dataset.repo_id="$REPO_ID" --dataset.root="$DATA_ROOT" --dataset.episodes="$TRAIN_EPISODES" --dataset.video_backend=pyav --dataset.use_imagenet_stats=true --dataset.image_transforms.enable=false --policy.chunk_size=100 --policy.n_action_steps=1 --policy.gripper_loss_weight=3.0 --policy.optimizer_lr_backbone=0.0001 --seed=1 --batch_size=4 --num_workers=4 --steps="$STEPS" --log_freq="$LOG_FREQ" --save_freq="$SAVE_FREQ" --eval_freq=0 --wandb.enable=false --output_dir="$OUT" "${extra[@]}")
printf '%q ' "${cmd[@]}" > "$RECORDS/command.sh"
printf '\n' >> "$RECORDS/command.sh"
"${cmd[@]}" 2>&1 | tee "$RECORDS/train.log"
printf 'exit=0\n' > "$RECORDS/train.done"
date -Is > "$RECORDS/finished_at.txt"
