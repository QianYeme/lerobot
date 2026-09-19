# M-VF-ACT Conda 环境配置指南

> 从零开始，用 conda 搭建开发环境和训练环境

---

## 1. 总体架构

| 机器 | OS | Conda 环境 | Python | 用途 |
|------|-----|-----------|--------|------|
| 开发机 | Windows 11 | `lerobot-dev` | 3.12 | 代码编辑、语法检查、数据集管理 |
| 训练机 | Ubuntu 22.04 | `lerobot` | 3.12 | 模型训练、SAM 2、评估推理 |

---

## 2. Windows 开发环境

### 2.1 安装 Miniconda

```powershell
# 下载 Miniconda (Windows 64-bit)
# https://docs.conda.io/en/latest/miniconda.html

# 安装后打开 Anaconda Prompt (miniconda3)
conda --version
```

### 2.2 创建 `lerobot-dev` 环境

```bash
conda create -n lerobot-dev python=3.12 -y
conda activate lerobot-dev
```

### 2.3 安装依赖

```bash
# 基础工具
pip install openpyxl numpy

# 注: 不在 Windows 上安装 torch/torchvision/torchcodec
# 仅用于代码编辑和数据集管理
```

### 2.4 验证

```bash
python -c "print('Python 3.12 OK')"
conda env list
# lerobot-dev 应在列表中
```

---

## 3. Ubuntu 训练环境

### 3.1 安装 Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# 安装路径: ~/miniconda3
# 初始化 conda: yes

source ~/.bashrc
conda --version
```

### 3.2 创建 `lerobot` 环境

```bash
conda create -n lerobot python=3.12 -y
conda activate lerobot
```

### 3.3 安装 CUDA Toolkit (conda 内置)

```bash
# 安装 CUDA 11.8 运行时（无需系统级 CUDA）
conda install -c nvidia cuda-toolkit=11.8 -y

# 验证
nvcc --version
# Cuda compilation tools, release 11.8
```

### 3.4 安装 PyTorch

```bash
# PyTorch 2.x + CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 验证
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
vram = torch.cuda.get_device_properties(0).total_mem / 1e9
print(f'VRAM: {vram:.1f} GB')
"
# 预期: CUDA available: True, GPU: NVIDIA GeForce RTX 4090
```

### 3.5 安装 LeRobot

```bash
# 进入项目目录
cd /path/to/lerobot-main

# 安装核心依赖
pip install -e .

# 验证
python -c "import lerobot; print('LeRobot OK')"
python -c "from lerobot.policies.act_det import ACTDetConfig; print('ACTDet OK')"
```

### 3.6 安装 SAM 2（仅 mask 生成阶段）

```bash
pip install segment-anything-2 opencv-python scipy

# 下载模型权重
mkdir -p ~/checkpoints
cd ~/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
wget https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2_hiera_l.yaml
```

### 3.7 安装 WandB（可选）

```bash
pip install wandb
wandb login
```

### 3.8 完整依赖清单

```bash
# 查看所有已安装的包
conda list

# 导出环境（方便复现）
conda env export -n lerobot > environment.yml
```

---

## 4. 环境总结

```bash
# Windows 开发机
conda activate lerobot-dev
# Python 3.12 + openpyxl + numpy

# Ubuntu 训练机
conda activate lerobot
# Python 3.12 + PyTorch 2.x(CUDA 11.8) + LeRobot + SAM 2 + WandB
```

| 包 | 版本 | 用途 |
|----|------|------|
| python | 3.12 | 基础运行时 |
| cuda-toolkit | 11.8 | GPU 加速 |
| torch | >=2.2.1 | 深度学习框架 |
| torchvision | >=0.21.0 | 视觉模型 (ResNet18) |
| lerobot | 3.0+ | 机器人学习框架 |
| segment-anything-2 | latest | SAM 2 mask 生成 |
| einops | >=0.8.0 | 张量操作 |
| wandb | >=0.24.0 | 实验跟踪 |
| numpy | >=2.0.0 | 数值计算 |
| opencv-python-headless | >=4.9.0 | 图像/视频处理 |
