#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
T3_B3="experiments/position/t3_joint_state_spatial/index1/result/runs/T3_B3_s1/train.done"
T3_P4="experiments/position/t3_joint_state_spatial/index1/result/runs/T3_P4_s1/train.done"
echo "N0 queue started; waiting for T3 B3/P4 completion markers."
while [[ ! -f "$T3_B3" || ! -f "$T3_P4" ]]; do
  date -Is
  sleep 60
done
echo "T3 complete; starting N0 candidates in frozen order R2 -> R1 -> R3."
bash scripts/train_n0_overnight_candidate.sh r2 train
bash scripts/train_n0_overnight_candidate.sh r1 train
bash scripts/train_n0_overnight_candidate.sh r3 train
echo "N0 queue complete."
