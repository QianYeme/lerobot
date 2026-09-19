# 创建 Python 3.12 环境

> 在现有 `lerobot` (3.10) 旁边新建 `lerobot312`，两者互不影响。
> 全部完成后切换默认环境，确认训练正常后再删旧的。

```bash
# ============================================================
# Step 1: 创建环境
# ============================================================
source /root/miniconda3/etc/profile.d/conda.sh
conda create -n lerobot312 python=3.12 -y

# ============================================================
# Step 2: 安装 PyTorch (CUDA 13.0)
# ============================================================
conda activate lerobot312
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# ============================================================
# Step 3: 安装 LeRobot (项目本身)
# ============================================================
cd /root/autodl-tmp/lerobot/lerobot-main
pip install -e .

# ============================================================
# Step 4: 安装 SAM 2 (从 GitHub，复用已有 torch)
# ============================================================
pip install --no-build-isolation git+https://github.com/facebookresearch/sam2.git

# ============================================================
# Step 5: 安装其余训练依赖
# ============================================================
pip install decord opencv-python scipy wandb

# ============================================================
# Step 6: 验证
# ============================================================
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "from sam2.build_sam import build_sam2_video_predictor; print('SAM2: OK')"
python -c "import decord; print('decord: OK')"
lerobot-train --help 2>&1 | head -5

# ============================================================
# Step 7 (可选): 确认训练能跑后删旧环境
# ============================================================
# conda remove -n lerobot --all -y
# conda clean --all -y

# ============================================================
# Step 8: 设为默认
# ============================================================
# echo 'conda activate lerobot312' >> ~/.bashrc
```

> **注意**: Step 7 是确认训练正常后再执行，给系统盘腾空间。旧环境约占用 ~10-15 GB。
