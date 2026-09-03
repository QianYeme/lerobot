#!/usr/bin/env bash
# ============================================================
#  ACTDet 模型 + 数据集 一键下载脚本（Hugging Face 国内镜像）
#  用法：  bash download_all.sh
#  说明：
#   - 断点续传：中断后重跑即可，已下载的文件自动跳过
#   - 公开仓库，无需登录 / token
#   - 总下载量约 9.3 GB（模型 3.2G + formal_A 3.0G + formal1_B 2.9G）
#   - 依赖新版 huggingface_hub(>=1.x) 的 `hf` 命令（`huggingface-cli` 已废弃）
# ============================================================
set -e

# ---- 国内镜像 ----
export HF_ENDPOINT=https://hf-mirror.com

# 新版高性能传输由 Xet 承担，默认已启用；若镜像下大文件报错/极慢，
# 可取消下面这行注释，改回经典 HTTP 下载（对镜像最稳）：
# export HF_HUB_DISABLE_XET=1

# ---- lerobot-main 根目录（按需修改）----
BASE="/home/lyj/lerobot"
cd "$BASE"

# ---- 检查 hf 命令是否可用 ----
if ! command -v hf >/dev/null 2>&1; then
    echo "==> 未检测到 hf 命令，正在安装 huggingface_hub ..."
    pip install -U "huggingface_hub[hf-xet]"
fi

# $1=仓库名  $2=目标相对目录  $3=仓库类型(model|dataset)
dl() {
    local repo="$1" dest="$2" rtype="${3:-model}"
    echo ""
    echo "==> 下载 $repo  ->  $dest  (type=$rtype)"
    hf download "QYyyyyyyy/$repo" --local-dir "$dest" --repo-type "$rtype"
}

echo "============================================================"
echo " [1/2] 下载数据集（2 个）"
echo "============================================================"
dl formal_A   数据集/formal_A   dataset
dl formal1_B  数据集/formal1_B  dataset

echo ""
echo "============================================================"
echo " [2/2] 下载模型（15 个）"
echo "============================================================"
dl 02-55-15_act_checkpointsE1        outputs/train/2026-08-07/02-55-15_act/checkpointsE1/last/pretrained_model        model
dl 02-56-09_act_det_checkpointsE2    outputs/train/2026-08-07/02-56-09_act_det/checkpointsE2/last/pretrained_model    model
dl 16-25-39_act_det_checkpointsE3    outputs/train/2026-08-07/16-25-39_act_det/checkpointsE3/last/pretrained_model    model
dl 16-26-04_act_det_checkpointsE3b   outputs/train/2026-08-07/16-26-04_act_det/checkpointsE3b/last/pretrained_model   model
dl 16-28-36_act_checkpointsE4        outputs/train/2026-08-07/16-28-36_act/checkpointsE4/last/pretrained_model        model
dl 01-17-03_act_det_checkpointsE5    outputs/train/2026-08-08/01-17-03_act_det/checkpointsE5/last/pretrained_model    model
dl 01-17-21_act_det_checkpointsE6    outputs/train/2026-08-08/01-17-21_act_det/checkpointsE6/last/pretrained_model    model
dl 01-17-41_act_det_checkpointsE6b   outputs/train/2026-08-08/01-17-41_act_det/checkpointsE6b/last/pretrained_model   model
dl 12-32-41_act_det_checkpointsE7    outputs/train/2026-08-08/12-32-41_act_det/checkpointsE7/last/pretrained_model    model
dl 00-03-00_act_det_checkpointsE7b   outputs/train/2026-08-09/00-03-00_act_det/checkpointsE7b/last/pretrained_model   model
dl 12-33-40_act_det_checkpointsE8    outputs/train/2026-08-08/12-33-40_act_det/checkpointsE8/last/pretrained_model    model
dl 12-34-49_act_det_checkpointsE9    outputs/train/2026-08-08/12-34-49_act_det/checkpointsE9/last/pretrained_model    model
dl 00-03-19_act_det_checkpointsE_A0  outputs/train/2026-08-09/00-03-19_act_det/checkpointsE_A0/last/pretrained_model  model
dl 00-03-39_act_det_checkpointsE_A1  outputs/train/2026-08-09/00-03-39_act_det/checkpointsE_A1/last/pretrained_model  model
dl 09-02-51_act_det_checkpointsE_A2  outputs/train/2026-08-09/09-02-51_act_det/checkpointsE_A2/last/pretrained_model  model

echo ""
echo "============================================================"
echo " 全部下载完成。开始校验..."
echo "============================================================"

fail=0
for d in outputs/train/*/*/checkpointsE*/last/pretrained_model; do
    if [ -f "$d/model.safetensors" ] && [ -f "$d/config.json" ]; then
        echo "OK   $d"
    else
        echo "FAIL $d"
        fail=1
    fi
done
for d in 数据集/formal_A 数据集/formal1_B; do
    if [ -f "$d/meta/info.json" ]; then
        echo "OK   $d"
    else
        echo "FAIL $d（缺 meta/info.json）"
        fail=1
    fi
done
[ "$fail" = "0" ] && echo "全部校验通过 ✅" || echo "存在失败项，请重跑本脚本续传 ❌"
