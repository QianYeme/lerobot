# M-VF-ACT: 增强视觉感知的ACT算法在力控抓取任务中的应用

> 基于论文《增强视觉感知的ACT算法在机械臂装配任务研究》（陈绮颖，2026）
>
> 任务场景：力控抓取半满透明塑料杯（SO-ARM101 机械臂，LeRobot v3.0）

---

## 目录

1. [相对于论文的创新](#1-相对于论文的创新)
2. [完整架构](#2-完整架构)
3. [损失函数](#3-损失函数)
4. [推理架构](#4-推理架构)
5. [版本切换](#5-版本切换)
6. [操作指南](#6-操作指南)

---

## 1. 相对于论文的创新

### 论文已有功能（全部实现）

| 模块              | 论文章节      | 作用                                                                        |
| ----------------- | ------------- | --------------------------------------------------------------------------- |
| 在线数据增强      | §3.2         | 色彩抖动、高斯噪声、随机遮挡——仅对全局视角（top）图像施加，独立概率 p=0.9 |
| 目标检测分支      | §3.3.1-3.3.4 | 共享 ResNet18 + FPN + FCOS 检测头，Focal Loss + L1 回归 + BCE 中心度损失    |
| 检测-动作特征融合 | §3.3.3       | FPN 多尺度特征生成空间注意力图，注入动作分支                                |

### 本项目新增创新

| 创新编号         | 模块                                                   | 状态      | 作用                                                                                                                 |
| ---------------- | ------------------------------------------------------ | --------- | -------------------------------------------------------------------------------------------------------------------- |
| **创新 1** | **Mask-Guided Perception（SAM 2 掩膜引导感知）** | ✅ 已实现 | SAM 2 离线生成透明杯 mask → Mask Decoder 辅助训练 → 像素级分割损失梯度塑造 backbone，使模型从 RGB 中"看见"透明物体 |
| 创新 2           | Visual-Force Fusion（视觉-力感 Cross-Attention 融合）  | 📋 规划中 | 夹爪负载/电流作为 force token 与视觉特征在 Transformer 中融合，解决透明杯"碰没碰到"感觉不到的问题                    |
| 创新 3           | Hybrid Action Head（位置-力控双路 Action Head）        | 📋 规划中 | 预测关节位置 + 夹爪力控参数，适应不同水量下的稳定抓取                                                                |
| 创新 4           | Temporal Modeling（时序帧堆叠）                        | 📋 规划中 | 连续帧堆叠捕获水晃动动力学，防洒液                                                                                   |

### 创新 1 的设计思路

**透明塑料杯的核心问题**：RGB 图像中杯子几乎不可见——和灰色桌面融为一体。

**解决方案**：训练时，SAM 2 利用 CVAT 标注的第一帧 bbox 作为 prompt，对整段视频生成逐帧 mask（杯子区域=1，背景=0，边缘高斯模糊）。Mask Decoder 挂在 FPN 输出上，用逐像素 L1 loss 监督，梯度反向传播到共享 backbone。Mask Decoder 和 FCOS 检测头地位完全平等——都是训练时的辅助监督信号。

**推理时**：Mask Decoder 不运行。但 backbone 已在训练中被像素级 mask 梯度"雕刻"过，对透明物体的边缘、轮廓、区域比标准 ResNet 敏感得多。

---

## 2. 完整架构

### 2.1 训练架构

```
输入:
  observation.images.top    (B, 3, 480, 640)  全局视角 RGB
  observation.images.wrist  (B, 3, 480, 640)  腕部视角 RGB
  observation.state         (B, 6 或 9)        关节位置（+力感）
  action                    (B, 100, 6)        目标动作序列
  episode_index / frame_index                 用于查标注和 mask

                   ┌─────────────┐
                   │  Data Aug   │ ← 仅 top，在线随机
                   │  (色彩抖动   │
                   │   噪声/遮挡) │
                   └──────┬──────┘
                          ▼
┌──────────────────────────────────────────────────┐
│               Shared ResNet18                     │
│                                                   │
│  layer2 → F2 (128ch, 60×80)                       │
│  layer3 → F3 (256ch, 30×40)                       │
│  layer4 → F4 (512ch, 15×20)                       │
└──────────────────────┬───────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │   (top 视角，检测+Mask)    │    (wrist 视角，标准ACT)
         │                           │
         ▼                           │
   ┌──────────┐                      │
   │   FPN    │  特征金字塔           │
   └────┬─────┘                      │
        │                            │
   ┌────┼────┬──────────┐            │
   ▼    ▼    ▼          ▼            │
  P2   P3   P4    Mask Decoder       │
 (60) (30) (15)     (训练)           │
  ×80  ×40  ×20       │              │
   │    │    │         ▼              │
   │    │    │    pred_mask          │
   │    │    │    (480×640)          │
   │    │    │      │                │
   │    │    │      ▼                │
   │    │    │   Mask Loss ←── SAM2 GT mask
   │    │    │      │                │
   ▼    ▼    ▼      │ (梯度反向传播   │
  FCOS Head         │  塑造backbone)  │
    │                │                │
    ▼                │                │
 Detection Loss      │                │
 (Focal+L1+BCE)      │                │
    │                │                │
    └────────┬───────┘                │
             ▼                        │
   ┌────────────────────┐             │
   │  Detection-Feature │             │
   │  Fusion Module     │             │
   │  (空间注意力注入)   │             │
   └─────────┬──────────┘             │
             │                        │
             ▼                        ▼
      enhanced_f4                  f4 (原始)
      (512ch, 15×20)           (512ch, 15×20)
             │                        │
             └────────┬───────────────┘
                      ▼
            ┌──────────────────┐
            │ 1×1 Conv Project │
            │ 512 → dim_model  │
            │ flatten → tokens │
            └────────┬─────────┘
                     ▼
┌──────────────────────────────────────────────────┐
│            Transformer Encoder                    │
│  [latent | robot_state | img_tokens...]           │
│  + 1D/2D sinusoidal positional embeddings        │
└──────────────────────┬───────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────┐
│            Transformer Decoder                    │
│  learned queries → cross-attn → 逐层解码         │
└──────────────────────┬───────────────────────────┘
                       ▼
              ┌─────────────────┐
              │  Action Head    │
              │  Linear → (6,)  │
              └─────────────────┘
                       │
                       ▼
                Action_L1 Loss
```

### 2.2 文件结构

```
src/lerobot/policies/act_det/
├── __init__.py                     # 导出
├── configuration_act_det.py        # 所有配置参数（~120行）
├── modeling_act_det.py             # ACTDetPolicy + ACTDetModel（~530行）
├── processor_act_det.py            # 复用 ACT 处理器
├── label_loader.py                 # CVAT XML 标注加载
├── mask_loader.py                  # NPZ mask 加载（LRU 缓存）
└── detection/
    ├── __init__.py
    ├── fpn.py                      # Feature Pyramid Network
    ├── fcos.py                     # FCOS 检测头 + Focal/L1/BCE 损失
    ├── fusion.py                   # 检测-动作特征融合模块
    ├── mask_decoder.py             # Mask Decoder（FPN → 480×640 mask）
    └── augmentation.py             # 在线数据增强（色彩抖动/噪声/遮挡）

scripts/
└── generate_sam2_masks.py          # SAM 2 离线 mask 生成脚本
```

---

## 3. 损失函数

### 3.1 总损失

```
Total = L_action + L_KL + λ_det · L_detection + λ_mask · L_mask
```

| 超参数          | 默认值 | 说明                                                |
| --------------- | ------ | --------------------------------------------------- |
| `kl_weight`   | 10.0   | KL 散度权重（CVAE 正则化）                          |
| `det_weight`  | 10.0   | 检测损失权重（数值范围 ~0.01-0.1，需放大）          |
| `mask_weight` | 1.0    | Mask 损失权重（数值范围 ~0.5，与 action loss 相近） |

### 3.2 各损失项

**动作预测损失：**

```
L_action = |action_GT - action_pred|₁    (L1 损失，动作序列)
```

**KL 散度损失（CVAE）：**

```
L_KL = -½ Σ(1 + log σ² - μ² - σ²)
```

**检测损失：**

```
L_detection = L_cls + L_reg + L_ctr

L_cls = Focal Loss(α=0.25, γ=2.0)    # 分类：杯/背景
L_reg = |reg_GT - reg_pred|₁          # 回归：到四条边的距离 (l,t,r,b)
L_ctr = BCE(centerness_GT, ctr_pred)  # 中心度：抑制低质量检测
```

**Mask 损失（创新 1）：**

```
L_mask = |pred_mask - SAM2_GT_mask|₁  (逐像素 L1，仅 top 视角)
```

### 3.3 损失计算范围

| 损失项      | 训练                          | 推理      |
| ----------- | ----------------------------- | --------- |
| L_action    | ✅ 全部帧                     | ✅        |
| L_KL        | ✅                            | ❌（z=0） |
| L_detection | ✅ 有标注的帧                 | ❌        |
| L_mask      | ✅ 有 NPZ mask 的帧（仅 top） | ❌        |

---

## 4. 推理架构

```
输入:
  observation.images.top    (1, 3, 480, 640)
  observation.images.wrist  (1, 3, 480, 640)
  observation.state         (1, 6 或 9)
  latent                    全零向量 (1, 32)

              ┌───────────────────────────────┐
              │       Shared ResNet18          │
              │  layer2 → F2, layer3 → F3,     │
              │  layer4 → F4                   │
              └──────────────┬────────────────┘
                             │
       ┌─────────────────────┴─────────────────────┐
       │  top: FPN → Fusion → enhanced_f4           │
       │  wrist: F4 → 直接投影                       │
       │  (FCOS Head ✗)  (Mask Decoder ✗)         │
       └─────────────────────┬─────────────────────┘
                             ▼
              ┌─────────────────────────┐
              │  Transformer Encoder     │
              │  → Decoder → Action Head │
              └─────────────┬───────────┘
                            ▼
              ┌─────────────────────────┐
              │  action: (100, 6)        │
              │  关节目标位置序列         │
              └─────────────────────────┘
```

**关键点：**

- FCOS Head 和 Mask Decoder 均不运行——它们在训练时塑造了 backbone，推理时不需要。
- FPN 和 Fusion 模块在推理时运行——它们已经是 backbone 的一部分，权重已被检测+mask 梯度训练过。
- Latent 直接填零（CVAE 先验均值）。

---

## 5. 版本切换

通过 `use_detection` 和 `use_mask_guidance` 两个参数实现三版本切换：

| 版本                    | 策略类型                | 配置                                                |
| ----------------------- | ----------------------- | --------------------------------------------------- |
| **V1: 标准 ACT**  | `policy.type=act`     | 原始 ACT，无 FPN/Fusion/检测/Mask                   |
| **V2: 论文版**    | `policy.type=act_det` | `use_detection=true`, `use_mask_guidance=false` |
| **V3: 论文+Mask** | `policy.type=act_det` | `use_detection=true`, `use_mask_guidance=true`  |

**推理架构分类：**

- V1 → `Backbone(layer4) → Transformer`
- V2 和 V3 → `Backbone(layer2/3/4) → FPN → Fusion → Transformer`（相同推理路径，权重质量不同）

---

## 6. 操作指南

### 6.1 环境要求

| 环境    | 配置                                       |
| ------- | ------------------------------------------ |
| OS      | Ubuntu 22.04 LTS                           |
| Python  | ≥ 3.12                                    |
| GPU     | NVIDIA RTX 4090（16GB 显存）               |
| PyTorch | ≥ 2.2.1                                   |
| LeRobot | v3.0+                                      |
| SAM 2   | segment-anything-2（仅 mask 生成脚本需要） |

### 6.2 安装

```bash
# 1. 安装 LeRobot 及其依赖
cd lerobot-main
pip install -e .

# 2. 安装 SAM 2（仅在 Ubuntu 上，仅 mask 生成阶段）
pip install segment-anything-2 opencv-python scipy
```

### 6.3 数据集准备

**目录结构：**

```
dataset_root/
├── meta/
│   ├── info.json
│   ├── stats.json
│   └── episodes/chunk-000/file-000.parquet
├── data/
│   └── chunk-000/file-000.parquet
├── videos/
│   ├── observation.images.top/chunk-000/file-000.mp4
│   └── observation.images.gripper/chunk-000/file-000.mp4
└── annotations/
    ├── top/
    │   ├── episode_000.xml       ← CVAT 1.1 导出
    │   └── ...
    ├── gripper/
    │   ├── episode_000.xml
    │   └── ...
    └── masks/                    ← 由 SAM 2 脚本生成
        └── top/
            ├── episode_000.npz
            └── ...
```

**步骤：**

```bash
# Step 1: 确保 CVAT XML 标注文件已放在 annotations/{top,gripper}/ 下
# Step 2: 生成 SAM 2 mask
python scripts/generate_sam2_masks.py \
    --dataset_root /path/to/dataset_root \
    --annotation_dir annotations \
    --cameras top \
    --gaussian_sigma 2.0

# Step 3: 验证 mask 文件
ls annotations/masks/top/episode_*.npz | wc -l    # 应等于 episode 数
```

### 6.4 训练

```bash
# V1: 标准 ACT 基线
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=your_dataset \
    --training.steps=100000 \
    --training.batch_size=8

# V2: 论文版（检测）
lerobot-train \
    --policy.type=act_det \
    --policy.use_detection=true \
    --policy.use_mask_guidance=false \
    --policy.annotation_dir=/path/to/dataset/annotations \
    --policy.aug_enable=true \
    --dataset.repo_id=your_dataset \
    --training.steps=100000 \
    --training.batch_size=8

# V3: 论文 + Mask（默认 Top 检测 + Top Mask）
lerobot-train \
    --policy.type=act_det \
    --policy.use_detection=true \
    --policy.use_mask_guidance=true \
    --policy.annotation_dir=/path/to/dataset/annotations \
    --policy.aug_enable=true \
    --policy.det_cameras '{"observation.images.top": {"enable": true}, "observation.images.wrist": {"enable": false}}' \
    --policy.mask_cameras '{"observation.images.top": {"enable": true}, "observation.images.wrist": {"enable": false}}' \
    --dataset.repo_id=your_dataset \
    --training.steps=100000 \
    --training.batch_size=8

# V3+: 双视角检测 + 双视角 Mask（全开）
lerobot-train \
    --policy.type=act_det \
    --policy.use_detection=true \
    --policy.use_mask_guidance=true \
    --policy.det_cameras '{"observation.images.top": {"enable": true}, "observation.images.wrist": {"enable": true}}' \
    --policy.mask_cameras '{"observation.images.top": {"enable": true}, "observation.images.wrist": {"enable": true}}' \
    --policy.annotation_dir=/path/to/dataset/annotations \
    --policy.aug_enable=true \
    --dataset.repo_id=your_dataset \
    --training.steps=100000 \
    --training.batch_size=8
    --training.batch_size=8
```

### 6.5 评估

```bash
lerobot-eval \
    --policy.type=act_det \
    --policy.pretrained_path=/path/to/checkpoint \
    --dataset.repo_id=your_dataset \
    --eval.episodes=30
```

### 6.6 消融实验

**数据增强消融：**

```yaml
# 仅色彩抖动
aug_noise_enable: false
aug_occlusion_enable: false

# 仅噪声（论文最佳）
aug_color_jitter_enable: false
aug_occlusion_enable: false

# 仅遮挡
aug_color_jitter_enable: false
aug_noise_enable: false
```

**检测视角消融：**

```yaml
# Top 检测 + Wrist 检测
det_cameras:
  observation.images.top:   {enable: true}
  observation.images.wrist: {enable: true}

# 仅 Top 检测
det_cameras:
  observation.images.top:   {enable: true}
  observation.images.wrist: {enable: false}
```

**Mask 消融：**

```yaml
# 关闭 Mask（等价 V2 论文版）
use_mask_guidance: false
```

**Mask 视角消融：**

```yaml
# Top Mask + Wrist Mask（双视角Mask）
mask_cameras:
  observation.images.top:   {enable: true}
  observation.images.wrist: {enable: true}

# 仅 Top Mask（默认）
mask_cameras:
  observation.images.top:   {enable: true}
  observation.images.wrist: {enable: false}

# 仅 Wrist Mask
mask_cameras:
  observation.images.top:   {enable: false}
  observation.images.wrist: {enable: true}
```

**Mask 权重消融：**

```yaml
# 调大 Mask 损失贡献
mask_weight: 5.0
# 调小
mask_weight: 0.5
```

### 6.7 关键配置参数速查

| 参数                        | 默认值   | 说明                   |
| --------------------------- | -------- | ---------------------- |
| `chunk_size`              | 100      | 一次预测的动作序列长度 |
| `n_action_steps`          | 100      | 实际执行的动作步数     |
| `dim_model`               | 512      | Transformer 隐藏维度   |
| `latent_dim`              | 32       | VAE 潜在空间维度       |
| `use_vae`                 | true     | CVAE 开关              |
| `kl_weight`               | 10.0     | KL 散度权重            |
| `use_detection`           | true     | 检测分支开关           |
| `det_weight`              | 10.0     | 检测损失权重           |
| `det_cameras`             | top:true, wrist:false | 检测每视角独立开关 |

| `fcos_num_classes`        | 1        | 检测类别数（cup）      |
| `focal_alpha`             | 0.25     | Focal Loss α          |
| `focal_gamma`             | 2.0      | Focal Loss γ          |
| `use_mask_guidance`       | true     | Mask 引导开关          |
| `mask_weight`             | 1.0      | Mask 损失权重          |
| `mask_cameras`            | top:true, wrist:false | Mask 每视角独立开关 |

| `aug_enable`              | true     | 数据增强开关           |
| `aug_probability`         | 0.9      | 每方法独立应用概率     |
| `aug_color_jitter_enable` | true     | 色彩抖动               |
| `aug_noise_enable`        | true     | 高斯噪声               |
| `aug_occlusion_enable`    | true     | 随机遮挡               |
| `optimizer_lr`            | 1e-5     | 学习率                 |
| `optimizer_lr_backbone`   | 1e-5     | Backbone 学习率        |
| `vision_backbone`         | resnet18 | 视觉骨干网络           |

### 6.8 模型参数量

| 版本               | 参数量      |
| ------------------ | ----------- |
| V1: 标准 ACT       | ~52M        |
| V2: 论文版（检测） | ~58M        |
| V3: 论文+Mask      | ~58M + ~20K |

---

## 7. 快速使用说明

### 7.1 三版本一键命令

```bash
# 标准ACT
lerobot-train --policy.type=act

# 论文版（检测，Top视角检测 + 无Mask）
lerobot-train --policy.type=act_det --policy.use_mask_guidance=false

# 检测+Mask（Top视角检测 + Top视角Mask，默认）
lerobot-train --policy.type=act_det --policy.use_mask_guidance=true

# 双视角全开（Top+Wrist 检测 + Top+Wrist Mask）
lerobot-train --policy.type=act_det \
    --policy.use_mask_guidance=true \
    --policy.det_cameras '{"observation.images.top":{"enable":true},"observation.images.wrist":{"enable":true}}' \
    --policy.mask_cameras '{"observation.images.top":{"enable":true},"observation.images.wrist":{"enable":true}}'
```

### 7.2 在 Ubuntu 运行前需要做的

```bash
# 1. 安装 SAM 2（仅 mask 生成阶段需要）
pip install segment-anything-2

# 2. 运行 SAM 2 离线生成 mask
python scripts/generate_sam2_masks.py --dataset_root /path/to/dataset

# 3. 确认 annotations/masks/top/ 下有所有 episode 的 .npz 文件
ls annotations/masks/top/episode_*.npz | wc -l

# 4. 安装 LeRobot 并正常训练
pip install -e .
lerobot-train --policy.type=act_det --policy.annotation_dir=/path/to/dataset/annotations
```

---

## 8. 训练集文件格式

### 8.1 目录结构

```
dataset_root/
├── meta/
│   ├── info.json                          # 数据集元信息（episode数、帧数、特征定义）
│   ├── stats.json                         # 各特征的统计值（mean/std）
│   ├── tasks.parquet                      # 任务标签
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet           # episode 元数据

├── data/
│   └── chunk-000/
│       └── file-000.parquet               # 主数据表（action, state, timestamp, episode_index, frame_index）

├── videos/
│   ├── observation.images.top/
│   │   └── chunk-000/
│   │       └── file-000.mp4               # 全局视角视频（640×480, 30fps）
│   └── observation.images.gripper/
│       └── chunk-000/
│           └── file-000.mp4               # 腕部视角视频（640×480, 30fps）

└── annotations/                           # ← 模型额外需要的标注文件
    ├── top/
    │   ├── episode_000.xml                # CVAT 1.1 XML, top视角检测标注
    │   └── ...
    ├── gripper/
    │   ├── episode_000.xml                # CVAT 1.1 XML, wrist视角检测标注
    │   └── ...
    └── masks/
        └── top/
            ├── episode_000.npz            # SAM 2 离线生成的mask (N,480,640) float32
            └── ...
```

### 8.2 `meta/info.json` 关键字段

```json
{
    "codebase_version": "v3.0",
    "robot_type": "so_follower",
    "total_episodes": 180,
    "total_frames": 129240,
    "fps": 30,
    "splits": { "train": "0:180" },
    "features": {
        "action": {
            "dtype": "float32",
            "names": [
                "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"
            ],
            "shape": [6]
        },
        "observation.state": {
            "dtype": "float32",
            "names": [
                "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
                "gripper.load", "gripper.curr", "master_gripper.pos"
            ],
            "shape": [9]
        },
        "observation.images.top": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "info": { "video.fps": 30, "video.codec": "av1" }
        },
        "observation.images.gripper": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "info": { "video.fps": 30, "video.codec": "av1" }
        },
        "episode_index": { "dtype": "int64", "shape": [1] },
        "frame_index":   { "dtype": "int64", "shape": [1] },
        "timestamp":     { "dtype": "float32", "shape": [1] },
        "index":         { "dtype": "int64", "shape": [1] },
        "task_index":    { "dtype": "int64", "shape": [1] }
    }
}
```

### 8.3 `data/file-000.parquet` 字段说明

| 列名 | 类型 | 维度 | 说明 |
|------|------|------|------|
| `action` | float32 | (6,) | 目标关节位置（shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper） |
| `observation.state` | float32 | (6 或 9) | 当前关节位置+力感信号。原版数据 6 维，改进版 9 维（+load/curr/master_gripper） |
| `episode_index` | int64 | (1,) | 当前帧所属 episode（0 ~ total_episodes-1） |
| `frame_index` | int64 | (1,) | episode 内帧索引（0 ~ 717，约 23s×30fps） |
| `timestamp` | float32 | (1,) | 采集时间戳 |
| `index` | int64 | (1,) | 全局帧编号 |
| `task_index` | int64 | (1,) | 任务编号 |
| `action_is_pad` | bool | (1,) | 是否填充帧（训练时自动生成） |

### 8.4 CVAT 标注 XML 格式

```xml
<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <name>top_ep000</name>
      <size>718</size>
      <mode>interpolation</mode>
      <labels>
        <label><name>cup</name></label>
      </labels>
      <original_size><width>640</width><height>480</height></original_size>
    </task>
  </meta>
  <track id="0" label="cup" source="manual">
    <box frame="0" keyframe="1" outside="0" occluded="0"
         xtl="397.63" ytl="95.67" xbr="497.00" ybr="190.30" z_order="0"/>
    <box frame="1" keyframe="0" outside="0" occluded="0"
         xtl="397.59" ytl="95.51" xbr="496.96" ybr="190.14" z_order="0"/>
    ...
  </track>
</annotations>
```

- `frame="N"` → 对应 parquet 中的 `frame_index`
- 文件名 `episode_{index:03d}.xml` → 对应 `episode_index`
- `xtl/ytl/xbr/ybr` → 像素坐标 bbox，训练时转换为 FCOS 格式（`l, t, r, b` 距离）
- `label` → 单类 `cup`，FCOS 中 `num_classes=1`

### 8.5 SAM 2 Mask NPZ 格式

- 文件名：`episode_{index:03d}.npz`
- 内部数组：`masks`，shape `(N_frames, 480, 640)`，dtype `float32`，值域 `[0, 1]`
- 生成方式：SAM 2 + 第一帧 bbox prompt（从 XML 提取）→ 视频逐帧传播 → 高斯模糊边缘（σ=2.0）
- 加载方式：`MaskLoader` 按 `(camera, episode_index, frame_index)` 查表，LRU 缓存

### 8.6 训练时数据流

```
Parquet (episode_index, frame_index)  ──→  查 MP4 视频帧 (RGB)  ──→  Data Augmentation (仅top)
                                       │
                                       ├──→  查 XML (bbox)        ──→  FCOS Head → Detection Loss
                                       │
                                       └──→  查 NPZ (mask)        ──→  Mask Decoder → Mask Loss

action 标签从 Parquet 的 action 列直接读取 → Action L1 Loss
```
