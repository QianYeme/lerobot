#!/usr/bin/env bash
# 逐个跑 15 个模型的离线评估（顺序执行，避免 8GB 显存 OOM），结果写入 outputs/offline_eval_results/
set -u

cd /home/lyj/lerobot
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate lerobotnn

RESDIR="outputs/offline_eval_results"
mkdir -p "$RESDIR"

SCRIPT="src/lerobot/scripts/offline_eval_act_det.py"
EPISODES="63-89"
BATCH=2

# name<TAB>checkpoint<TAB>repo_id
ROWS=(
"E1|outputs/train/2026-08-07/02-55-15_act/checkpointsE1/last/pretrained_model|formal_A"
"E2|outputs/train/2026-08-07/02-56-09_act_det/checkpointsE2/last/pretrained_model|formal_A"
"E3|outputs/train/2026-08-07/16-25-39_act_det/checkpointsE3/last/pretrained_model|formal_A"
"E3b|outputs/train/2026-08-07/16-26-04_act_det/checkpointsE3b/last/pretrained_model|formal_A"
"E4|outputs/train/2026-08-07/16-28-36_act/checkpointsE4/last/pretrained_model|formal1_B"
"E5|outputs/train/2026-08-08/01-17-03_act_det/checkpointsE5/last/pretrained_model|formal1_B"
"E6|outputs/train/2026-08-08/01-17-21_act_det/checkpointsE6/last/pretrained_model|formal1_B"
"E6b|outputs/train/2026-08-08/01-17-41_act_det/checkpointsE6b/last/pretrained_model|formal1_B"
"E7|outputs/train/2026-08-08/12-32-41_act_det/checkpointsE7/last/pretrained_model|formal1_B"
"E7b|outputs/train/2026-08-09/00-03-00_act_det/checkpointsE7b/last/pretrained_model|formal1_B"
"E8|outputs/train/2026-08-08/12-33-40_act_det/checkpointsE8/last/pretrained_model|formal1_B"
"E9|outputs/train/2026-08-08/12-34-49_act_det/checkpointsE9/last/pretrained_model|formal1_B"
"E_A0|outputs/train/2026-08-09/00-03-19_act_det/checkpointsE_A0/last/pretrained_model|formal1_B"
"E_A1|outputs/train/2026-08-09/00-03-39_act_det/checkpointsE_A1/last/pretrained_model|formal1_B"
"E_A2|outputs/train/2026-08-09/09-02-51_act_det/checkpointsE_A2/last/pretrained_model|formal1_B"
)

echo "===== 离线评估开始 $(date '+%F %T') ====="
echo "共 ${#ROWS[@]} 个模型，episodes=$EPISODES，batch=$BATCH"
echo ""

i=0
for row in "${ROWS[@]}"; do
  i=$((i+1))
  name="${row%%|*}"; rest="${row#*|}"; ckpt="${rest%%|*}"; repo="${rest##*|}"
  out="$RESDIR/${name}.json"
  log="$RESDIR/${name}.log"
  t0=$(date +%s)
  echo "[$i/${#ROWS[@]}] $name  (repo=$repo)  开始 $(date '+%T')"
  python "$SCRIPT" \
      --checkpoint "$ckpt" \
      --dataset.repo_id "$repo" \
      --dataset.root "/home/lyj/lerobot/数据集/$repo" \
      --episodes "$EPISODES" \
      --batch-size "$BATCH" \
      --num-workers 0 \
      --output "$out" > "$log" 2>&1
  rc=$?
  t1=$(date +%s)
  if [ $rc -eq 0 ]; then
    echo "    ✅ 完成 用时 $(( (t1-t0)/60 ))m$(( (t1-t0)%60 ))s"
  else
    echo "    ❌ 失败 rc=$rc  用时 $(( (t1-t0)/60 ))m$(( (t1-t0)%60 ))s  (见 $log)"
  fi
done

echo ""
echo "===== 全部结束 $(date '+%F %T') ====="
