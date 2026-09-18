#!/usr/bin/env bash
# A finite three-episode label preview; no policy training or robot control.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
OUT=outputs/mask_inject_phase1_20260917
export PYTHONNOUSERSITE=1
export TQDM_DISABLE=1
export PYTHONPATH="/root/autodl-tmp/c50-mask-env/sam2-minimal:$PROJECT/src"
python_bin=/root/autodl-tmp/c50-mask-env/bin/python
"$python_bin" -m pip freeze > "$OUT/sam_environment.txt"
git -C /root/autodl-tmp/c50-mask-env/sam2-minimal rev-parse HEAD > "$OUT/sam_source_commit.txt"
"$python_bin" -u scripts/prepare_c50_mask_preview.py \
  --dataset-root "$PROJECT/数据集/formal1_C50_nomaster" \
  --output "$OUT/mask_preview" --episodes 0 20 49 \
  --checkpoint checkpoints/sam2.1_hiera_large.pt 2>&1 | tee "$OUT/mask_preview.log"
date -Is > "$OUT/mask_preview.finished_at.txt"
