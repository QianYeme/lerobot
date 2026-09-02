# ACTDet — 增强视觉感知的ACT算法 实现文档

> 基于论文《增强视觉感知的ACT算法在机械臂装配任务研究》（陈绮颖，2026）
>
> 实现日期：2026-06-17

---

## 1. 概述

ACTDet（Action Chunking Transformer with Detection）在原始 ACT 模型基础上添加了四个核心增强：

| 模块 | 作用 | 论文对应章节 |
|------|------|-------------|
| **在线数据增强** | 仅对全局视角（top）图像施加色彩抖动、高斯噪声、随机遮挡 | §3.2 |
| **目标检测分支** | 共享ResNet18 + FPN + FCOS检测头，联合训练空间感知 | §3.3.1-3.3.4 |
| **检测-动作特征融合** | FPN多尺度特征生成空间注意力图，引导动作分支聚焦关键区域 | §3.3.3 |
| **Mask引导感知** | SAM 2 预生成透明物体mask → Mask Decoder 辅助训练 → 增强backbone对透明物体的感知 | 创新1 |

训练总损失：
```
Total Loss = Action_L1 + KL_div + det_weight × Detection_Loss + mask_weight × Mask_L1_Loss
Detection_Loss = Focal_Loss + L1_Reg + BCE_Centerness
Mask_L1_Loss = |pred_mask - SAM2_GT_mask|.mean()
```

---

## 2. 文件结构

```
src/lerobot/policies/act_det/
├── __init__.py                        # 包入口
├── configuration_act_det.py           # ACTDetConfig 配置类
├── modeling_act_det.py                # ACTDetPolicy + ACTDetModel
├── processor_act_det.py               # 前/后处理器（复用 ACT）
├── label_loader.py                    # CVAT XML 标注加载器
└── detection/
    ├── __init__.py
    ├── fpn.py                         # Feature Pyramid Network
    ├── fcos.py                        # FCOS 检测头 + 损失函数
    ├── fusion.py                      # 检测-动作特征融合模块
    └── augmentation.py                # 在线数据增强
```

---

## 3. 环境准备

### 3.1 依赖

- Python ≥ 3.12
- PyTorch ≥ 2.2.1
- torchvision ≥ 0.21.0
- LeRobot v3.0+ 及其依赖

```bash
pip install -e .
```

### 3.2 数据集标注文件准备

CVAT 1.1 XML 标注文件需要按以下结构放在数据集根目录下：

```
dataset_root/
├── meta/
├── data/
├── videos/
│   ├── observation.images.top/
│   └── observation.images.gripper/
└── annotations/                       # ← 新增
    ├── top/
    │   ├── episode_025.xml            # episode 25 的 top 视角标注
    │   ├── episode_026.xml
    │   └── ...
    ├── gripper/
    │   ├── episode_025.xml            # episode 25 的 gripper 视角标注
    │   └── ...
    └── masks/                          # ← Mask引导感知需要
        └── top/
            ├── episode_025.npz         # SAM 2 生成的 mask (N,480,640) float32
            └── ...
```

**XML 文件命名规则**：`episode_{index:03d}.xml`，其中 `index` 与数据集 parquet 中的 `episode_index` 一一对应。

**XML 格式要求**：CVAT 1.1 导出格式，label 名称为实际类别名（如 `cup`）。

**NPZ 文件命名规则**：`episode_{index:03d}.npz`，包含单个数组 `masks`，shape=(N_frames, 480, 640)，dtype=float32，值域[0,1]。

### 3.3 Mask 标注生成流程

1. 在 Ubuntu 上安装 SAM 2：`pip install segment-anything-2`
2. 确保 CVAT XML 标注文件已放在 `annotations/{camera}/` 下
3. 运行生成脚本：
   ```bash
   python scripts/generate_sam2_masks.py \
       --dataset_root /path/to/dataset \
       --annotation_dir annotations \
       --cameras top
   ```
4. 脚本自动从 XML 提取第一帧 bbox 作为 SAM 2 prompt，逐帧生成 mask
5. 检查 `annotations/masks/top/` 下的 NPZ 文件数量和内容

### 3.4 标注生成流程（CVAT XML）

1. 在 CVAT 中创建任务，上传按 episode 拆分好的视频
2. 使用 interpolation 模式逐帧标注（或用 tracker 辅助）
3. 导出为 CVAT 1.1 XML 格式
4. 解压并放置到上述目录结构中

---

## 4. 配置参数说明

### 4.1 训练配置示例 (YAML)

```yaml
# config_act_det.yaml
policy:
  type: act_det
  chunk_size: 100
  n_action_steps: 100
  dim_model: 512
  latent_dim: 32
  use_vae: true
  kl_weight: 10.0

  # === 检测分支 ===
  use_detection: true
  det_weight: 10.0
  det_cameras:
    observation.images.top:
      enable: true           # top 视角参与检测
    observation.images.wrist:
      enable: false          # wrist 视角不参与检测

  # === 标注路径 ===
  annotation_dir: /path/to/dataset/annotations

  # === FPN 参数 ===
  fpn_channels: 128
  fpn_in_channels: [128, 256, 512]

  # === FCOS 参数 ===
  fcos_num_classes: 1        # cup 单类别
  fcos_strides: [8, 16, 32]
  fcos_size_ranges:
    - [0, 60]                # P2 负责
    - [60, 120]              # P3 负责
    - [120, 99999]           # P4 负责
  focal_alpha: 0.25
  focal_gamma: 2.0

  # === 特征融合 ===
  fusion_hidden: 64

  # === 数据增强 ===
  aug_enable: true
  aug_probability: 0.9

  aug_color_jitter_enable: true
  aug_brightness: [0.8, 1.2]
  aug_contrast: [0.8, 1.2]
  aug_saturation: [0.8, 1.2]
  aug_hue: [-0.1, 0.1]

  aug_noise_enable: true
  aug_noise_std_range: [0.01, 0.05]

  aug_occlusion_enable: true
  aug_occlusion_area_ratio: [0.1, 0.3]
  aug_occlusion_gray_range: [0.3, 0.7]

  # === Mask-Guided Perception ===
  use_mask_guidance: true
  mask_weight: 1.0
  mask_dir: null                  # null = {annotation_dir}/masks
  mask_decoder_channels: 32
```

### 4.2 关键参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_detection` | `true` | 检测分支总开关 |
| `det_weight` | `10.0` | 检测损失在总损失中的权重 |
| `det_cameras.{key}.enable` | top:true, wrist:false | 每视角检测开关 |
| `annotation_dir` | `None` | XML标注目录，`None`=退化标准ACT |
| `aug_enable` | `true` | 数据增强总开关 |
| `aug_probability` | `0.9` | 每种增强独立应用概率 |
| `fcos_num_classes` | `1` | 检测类别数（不含背景） |
| `focal_alpha` | `0.25` | Focal Loss α 参数 |
| `focal_gamma` | `2.0` | Focal Loss γ 参数 |
| `use_mask_guidance` | `true` | Mask引导感知总开关 |
| `mask_weight` | `1.0` | Mask 损失在总损失中的权重 |
| `mask_dir` | `None` | NPZ mask 目录，`None` = `{annotation_dir}/masks` |
| `mask_decoder_channels` | `32` | Mask Decoder 上采样路径中间通道数 |

---

## 5. 版本切换

通过两个 bool 参数实现三个版本的切换：

| 版本 | `use_detection` | `use_mask_guidance` | 命令 |
|------|-----------------|---------------------|------|
| **标准 ACT** | — | — | `--policy.type=act`（用原 `act` 包） |
| **论文版（检测）** | `true` | `false` | `--policy.type=act_det --policy.use_mask_guidance=false` |
| **检测+Mask** | `true` | `true` | `--policy.type=act_det --policy.use_mask_guidance=true` |

三个版本训练的模型，推理架构分为两类：
- 标准 ACT → `Backbone(layer4) → Transformer`
- 论文版 / 检测+Mask → `Backbone(layer2/3/4) → FPN → Fusion → Transformer`（推理架构相同）

---

## 6. 使用方法

### 5.1 训练

```bash
lerobot-train \
    --policy.type=act_det \
    --policy.annotation_dir=/path/to/dataset/annotations \
    --dataset.repo_id=your_dataset_name \
    --training.steps=100000 \
    --training.batch_size=8 \
    --training.save_freq=10000
```

### 5.2 推理/评估

```bash
lerobot-eval \
    --policy.type=act_det \
    --policy.pretrained_path=/path/to/checkpoint \
    --dataset.repo_id=your_dataset_name \
    --eval.episodes=30
```

### 5.3 横向对比实验配置

通过调整配置开关，可以轻松进行消融实验：

| 实验 | 配置改动 |
|------|----------|
| **标准 ACT 基线** | `use_detection: false`, `aug_enable: false` |
| **ACT + 数据增强** | `use_detection: false`, `aug_enable: true` |
| **ACT + 目标检测** | `use_detection: true`, `aug_enable: false` |
| **ACT + Top 检测 + 噪声** | `use_detection: true`, `aug_enable: true`, `aug_color_jitter_enable: false`, `aug_occlusion_enable: false` |
| **ACT + 双视角检测** | `det_cameras.*.enable: true` |
| **ACT + 全增强** | 所有 `aug_*_enable: true`, `det_cameras.top.enable: true` |

### 5.4 增强方法独立消融

```yaml
# 仅色彩抖动
aug_noise_enable: false
aug_occlusion_enable: false

# 仅噪声
aug_color_jitter_enable: false
aug_occlusion_enable: false

# 仅遮挡
aug_color_jitter_enable: false
aug_noise_enable: false
```

---

## 7. 架构设计要点

### 6.1 共享 Backbone

检测分支和动作分支共用同一个 ResNet18，通过 `IntermediateLayerGetter` 提取 layer2/layer3/layer4 三层特征。检测梯度反向传播到 backbone，使视觉特征同时具备目标感知和操作指导能力（论文核心创新）。

### 7.2 推理行为

| 组件 | 训练 | 推理 |
|------|------|------|
| ResNet18 backbone | ✅ | ✅ |
| FPN | ✅ | ✅ |
| FCOS 检测头 | ✅（算损失） | ❌（跳过） |
| Mask Decoder | ✅（算损失，仅top） | ❌（跳过） |
| 特征融合模块 | ✅（训练权重） | ✅（生成注意力图引导动作） |
| 数据增强 | ✅（在线随机） | ❌（关闭） |
| CVAE编码器 | ✅ | ❌（z=0） |

### 7.3 标注加载策略

- 构造时一次性解析所有 XML 到内存字典
- 训练时通过 `(camera_key, episode_index, frame_index)` O(1) 查表
- 无标注的 episode/帧自动返回 None，检测损失跳过该帧
- 内存占用 ≈ 180 episode × 718 帧 × 20 字节 ≈ 2.6 MB

### 7.4 数据增强范围

**仅对 top（全局前方）摄像头图像施加增强**，wrist（腕部）图像保持不变。原因：
- top 视角受光照、视角变化影响更大，增强提升鲁棒性
- wrist 视角提供近景操作细节，过度增强引入噪声会影响精度

### 7.5 Mask 引导感知

- Mask Decoder 和 FCOS 检测头平行，都挂在 FPN 输出上作为训练辅助监督
- 仅 top 视角参与 mask 训练，因为透明杯子在全局视角中"看不到"的问题最严重
- Mask Decoder 在推理时丢弃，backbone 已被 pixel-level mask 梯度"雕刻"
- SAM 2 离线预生成 mask（NPZ 压缩文件），训练时直接加载
- 每帧都参与 mask loss，包括杯子出界帧（视为天然数据增强）

---

## 8. 更新日志

### 2026-06-17 — Mask-Guided Perception 新增

**新增文件（3个）：**

| 文件 | 行数 | 说明 |
|------|------|------|
| `policies/act_det/detection/mask_decoder.py` | 100 | 轻量分割解码器，FPN 三层上采样融合 → 480×640逐像素mask |
| `policies/act_det/mask_loader.py` | 115 | NPZ mask加载器，LRU缓存（~8 episodes in memory） |
| `scripts/generate_sam2_masks.py` | 200 | SAM 2离线mask生成脚本（Ubuntu运行） |

**修改文件（3个）：**

| 文件 | 改动说明 |
|------|----------|
| `policies/act_det/detection/__init__.py` | +1行：导出 MaskDecoder |
| `policies/act_det/configuration_act_det.py` | +5行：新增 `use_mask_guidance`, `mask_weight`, `mask_dir`, `mask_decoder_channels`，更新 docstring |
| `policies/act_det/modeling_act_det.py` | +80行：初始化 MaskDecoder/MaskLoader，forward 中插入 mask 预测（top仅训练时），新增 `_load_mask_batch` 方法，ACTDetPolicy.forward 中加 mask loss |

**核心设计决策：**
- Mask Decoder 和 FCOS 检测头呈平行关系——都挂在 FPN 输出上，作为训练时的辅助监督
- 仅 top 视角参与 mask 训练，wrist 不参与
- 推理时 Mask Decoder 不运行，backbone 已被 mask loss 的梯度"雕刻"
- 通过 `use_mask_guidance` 参数可实现三版本切换（标准ACT / 论文版 / 检测+Mask）

### 2026-06-17 — 初始实现

**新增文件（10个）：**

| 文件 | 行数 | 说明 |
|------|------|------|
| `policies/act_det/__init__.py` | 4 | 包入口，导出三个核心类 |
| `policies/act_det/configuration_act_det.py` | 118 | ACTDetConfig，继承 ACTConfig，新增 30+ 个参数 |
| `policies/act_det/modeling_act_det.py` | 390 | ACTDetPolicy + ACTDetModel，核心模型组装 |
| `policies/act_det/processor_act_det.py` | 40 | 复用 ACT 的前后处理器 |
| `policies/act_det/label_loader.py` | 150 | CVAT 1.1 XML 解析 + 内存缓存 |
| `policies/act_det/detection/__init__.py` | 12 | detection 子包入口 |
| `policies/act_det/detection/fpn.py` | 82 | Feature Pyramid Network（横向连接+自顶向下+平滑） |
| `policies/act_det/detection/fcos.py` | 229 | FCOS 检测头（三塔4层卷积） + Focal/L1/BCE 损失 |
| `policies/act_det/detection/fusion.py` | 122 | 检测-动作特征融合（注意力图生成+特征增强） |
| `policies/act_det/detection/augmentation.py` | 163 | 在线数据增强（色彩抖动/噪声/遮挡，独立开关） |

**修改文件（2个，仅注册导入）：**

| 文件 | 改动 |
|------|------|
| `policies/__init__.py` | +2行：导入并导出 `ACTDetConfig` |
| `policies/factory.py` | +1行：导入 `ACTDetConfig` 使 factory 可发现 |

**未修改（采集脚本保持原样）：**
- `scripts/lerobot_record.py`
- `robots/so_follower/so_follower.py`
- `motors/feetech/feetech.py`
- `motors/feetech/tables.py`

---

## 8. 注意事项

1. **标注目录不是必需的**：`annotation_dir=None` 时模型自动退化为标准 ACT（检测损失为0，融合模块以零注意力运行），不影响训练。

2. **检测只对标注过的帧生效**：没有标注的帧检测损失为0，检测分支参数通过有标注帧的梯度更新。

3. **增强只影响训练数据流**：推理时增强自动关闭，不影响策略稳定性。

4. **建议在 Ubuntu 上运行训练**：Windows 下 torchcodec 等依赖不可用（pyproject.toml 已限制 `sys_platform != 'win32'`）。

5. **FPN 输入特征名称为 `f2/f3/f4`**：与标准 ACT 的 `feature_map`（仅对应 `layer4`）不同，检测分支需要三层输出。
