# M-VF-ACT: 增强视觉感知的ACT算法在力控抓取任务中的应用

> 基于论文《增强视觉感知的ACT算法在机械臂装配任务研究》（陈绮颖，2026）
>
> 任务场景：力控抓取半满透明塑料杯（SO-ARM101 机械臂，LeRobot v3.0）

---

## 目录

1. [相对于论文的创新](#1-相对于论文的创新)
2. [完整架构](#2-完整架构)（训练 + 推理 + 文件结构）
3. [损失函数](#3-损失函数)
4. [版本切换](#4-版本切换)
5. [操作指南](#5-操作指南)
6. [快速使用说明](#6-快速使用说明)

---

## 1. 相对于论文的创新

### 论文已有功能（全部实现）

| 模块              | 论文章节      | 作用                                                                        |
| ----------------- | ------------- | --------------------------------------------------------------------------- |
| 在线数据增强      | §3.2         | 色彩抖动、高斯噪声、随机遮挡——仅对全局视角（top）图像施加，独立概率 p=0.9 |
| 目标检测分支      | §3.3.1-3.3.4 | 共享 ResNet18 + FPN + FCOS 检测头，Focal Loss + L1 回归 + BCE 中心度损失    |
| 检测-动作特征融合 | §3.3.3       | FPN 多尺度特征生成空间注意力图，注入动作分支                                |

### 本项目新增创新

| 创新编号         | 模块                                                   | 状态      | 作用                                                                                                                         |
| ---------------- | ------------------------------------------------------ | --------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **创新 1** | **Mask-Guided Perception（SAM 2 掩膜引导感知）** | ✅ 已实现 | SAM 2 离线生成透明杯 mask → Mask Decoder 辅助训练 → 像素级分割损失梯度塑造 backbone，使模型从 RGB 中"看见"透明物体         |
| **创新 2** | **FCOS Feature Injection（检测特征显式注入）**   | ✅ 已实现 | FCOS 检测头的分类+回归中间特征（centerness 门控）投影为额外 Encoder token，让 Transformer 显式访问"杯子在哪、多大"的检测知识 |
| **创新 3** | **Mask Feature Injection（分割特征显式注入）**   | ✅ 已实现 | Mask Decoder 的三层融合中间特征 f432 投影+降采样为额外 Encoder token，让 Transformer 显式访问"透明物体精细边缘"的分割知识    |
| 创新 4           | Visual-Force Fusion（视觉-力感 Cross-Attention 融合）  | 📋 规划中 | 夹爪负载/电流作为 force token 与视觉特征在 Transformer 中融合，解决透明杯"碰没碰到"感觉不到的问题                            |
| 创新 5           | Hybrid Action Head（位置-力控双路 Action Head）        | 📋 规划中 | 预测关节位置 + 夹爪力控参数，适应不同水量下的稳定抓取                                                                        |
| 创新 6           | Temporal Modeling（时序帧堆叠）                        | 📋 规划中 | 连续帧堆叠捕获水晃动动力学，防洒液                                                                                           |

### 创新 1 的设计思路

**透明塑料杯的核心问题**：RGB 图像中杯子几乎不可见——和灰色桌面融为一体。

**解决方案**：训练时，SAM 2 利用 CVAT 标注的第一帧 bbox 作为 prompt，对整段视频生成逐帧 mask（杯子区域=1，背景=0，边缘高斯模糊）。Mask Decoder 挂在 FPN 输出上，用逐像素 L1 loss 监督，梯度反向传播到共享 backbone。Mask Decoder 和 FCOS 检测头地位完全平等——都是训练时的辅助监督信号。

**推理时**：Mask Decoder 不运行。但 backbone 已在训练中被像素级 mask 梯度"雕刻"过，对透明物体的边缘、轮廓、区域比标准 ResNet 敏感得多。

### 创新 2 的设计思路：FCOS 特征注入

**核心问题**：创新 1 的 Mask Decoder 和创新 2 的 FCOS 检测头对策略网络的贡献只有「梯度反传」一条路径——它们的输出（分类分数、bbox、mask）仅用于计算 loss，不进入 Transformer Encoder。这意味着 Transformer 从未「看到」检测或分割的结果，只能间接依赖被梯度雕刻过的 backbone 特征。

**解决方案**：将 FCOS 检测头的**中间 tower 特征**（在最终预测层之前）投影为与图像 token 同维度的额外 token，直接拼入 Transformer Encoder 序列。

**具体做法：**

1. 取 FCOS 三个并行 tower 中的两个——`cls_feature`（分类特征，128 维）和 `reg_feature`（回归特征，128 维）——在 tower 输出之后、最终预测层（1×1 Conv → 1/4 通道）之前截取
2. 通道维拼接得到 256 维特征，用 centerness 预测值做空间门控：`combined × (1.0 + sigmoid(ctr_pred))`——高置信度区域特征放大到 2 倍，低置信度区域保持原样
3. 1×1 Conv 投影 256 → dim_model（512），得到与图像 token 同形状的 `(B, 512, H, W)`
4. 添加与同视角图像 token 相同的 2D 正弦位置编码后 flatten，紧跟在图像 token 之后拼入 Encoder

**默认只注入 P4 层级**（15×20 = 300 tokens），可通过 `fcos_inject_levels` 参数扩展到 P3、P2。

**与纯梯度反传的对比：**

| 机制                             | 创新 1（纯梯度）                       | 创新 2（显式注入）                   |
| -------------------------------- | -------------------------------------- | ------------------------------------ |
| 检测信息如何到达策略             | 梯度 → FPN → Backbone → 间接影响 F4 | 中间特征 → 投影 → 直接作为 token   |
| 推理时是否运行                   | 不运行                                 | 运行（tower 前向）                   |
| Transformer 能否显式关注检测区域 | 否                                     | 是（self-attention 可跨 token 关联） |
| 额外推理开销                     | 零                                     | ~3ms（P4 tower）                     |

**配置开关：** `fcos_feature_inject=True`（默认 False），`fcos_inject_levels=["p4"]`

### 创新 3 的设计思路：Mask 特征注入

**核心问题**：与创新 2 相同——Mask Decoder 的中间特征包含丰富的"透明物体边缘/轮廓"信息，但从未显式进入 Transformer。纯梯度反传可能不足以将精细的边缘语义传递到策略网络。

**解决方案/**：取 Mask Decoder 的**三层融合中间特征 f432**（P2/P3/P4 完全融合后、×8 上采样之前），投影 + 降采样后作为额外 token 拼入 Encoder。

**具体做法：**

1. 取 `_compute_f432(p2, p3, p4)` 的输出——`(B, 32, 60, 80)`，这是 P2/P3/P4 三层 FPN 特征经 reduce → upsample → concat → fuse 两轮融合后的 32 维压缩语义
2. 1×1 Conv 投影 32 → dim_model（512），得到 `(B, 512, 60, 80)`
3. adaptive_avg_pool2d 降采样到 `mask_inject_pool_size`（默认 15×20，300 tokens）——避免 token 数爆炸（60×80 = 4800 tokens）
4. 添加 2D 正弦位置编码后 flatten，紧跟图像 token（和 FCOS 注入 token）之后拼入 Encoder

**为什么注入 f432 而不注入最终 pred_mask：**

- 最终 pred_mask 是 1 通道 × 480×640 = 307,200 像素——全部 flatten 会炸掉 token 数
- f432 是 32 维、60×80 的浓缩语义——包含多尺度信息但空间分辨率可控
- 32 个通道各自编码了不同的语义模式（边缘方向、纹理类型、区域归属等），比单通道 mask 信息丰富得多

**配置开关：** `mask_feature_inject=True`（默认 False），`mask_inject_pool_size=(15, 20)`

---

## 2. 完整架构

> 以下以 **V6（全注入）** 为例展示完整数据流。图中标注了每条路径的「生命周期」：
>
> - `[训练 only]` — 仅训练时运行，推理时丢弃
> - `[训练+推理]` — 训练和推理都运行
> - `[仅用于 loss]` — 输出只参与损失计算，不进入 Encoder

### 2.1 训练架构（V6 全注入）

```
                          ┌──────────────────────────────┐
                          │         输  入  层            │
                          │                              │
                          │  top RGB     (B, 3, 480, 640)│
                          │  wrist RGB   (B, 3, 480, 640)│
                          │  robot state (B, 6 或 9)      │
                          │  action GT   (B, 100, 6)      │
                          │  episode_idx / frame_idx      │
                          └─────────────┬────────────────┘
                                        │
                          ┌─────────────▼────────────────┐
                          │   Data Augmentation           │  [训练 only，仅 top 视角]
                          │   色彩抖动 / 高斯噪声 / 遮挡   │
                          └─────────────┬────────────────┘
                                        │
          ┌─────────────────────────────▼───────────────────────────┐
          │                 Shared ResNet18 Backbone                │  [训练+推理]
          │  layer1 → ··· → layer2 → F2 (128, 60×80)               │
          │                  layer3 → F3 (256, 30×40)               │
          │                  layer4 → F4 (512, 15×20)               │
          └──────────┬──────────────────────────────────┬───────────┘
                     │                                  │
       ┌─────────────▼──────────┐            ┌──────────▼──────────┐
       │   top 视角 (检测+Mask)  │            │  wrist 视角 (标准ACT)│
       │            FPN          │            │                     │
       │       P2  P3  P4        │            │     F4 (原始)       │
       └──┬───────┬───────┬──────┘            └──────────┬──────────┘
          │       │       │                              │
          │       │       │                              │
    ┌─────▼──┐ ┌──▼──┐ ┌──▼──────────┐                  │
    │  P2    │ │ P3  │ │    P4        │                  │
    │60×80   │ │30×40│ │  15×20       │                  │
    └──┬──┬──┘ └──┬──┘ └──┬──┬──┬────┘                  │
       │  │       │        │  │  │                       │
       │  │       │        │  │  │                       │
       │  │       │   ┌────┘  │  └─────────┐             │
       │  │       │   │       │            │             │
       ▼  ▼       ▼   ▼       ▼            ▼             │
  ┌──────────────────────────────────────────────┐       │
  │              FCOS Head                        │       │
  │                                              │       │
  │  ┌──────────┐ ┌──────────┐ ┌──────────┐     │       │
  │  │cls_tower │ │reg_tower │ │ctr_tower │     │       │
  │  │ (4×Conv) │ │ (4×Conv) │ │ (4×Conv) │     │       │
  │  └────┬─────┘ └────┬─────┘ └────┬─────┘     │       │
  │       │             │            │           │       │
  │       │   [训练 only — 用于 loss] │           │       │
  │       ▼             ▼            ▼           │       │
  │  cls_logits    reg_pred     ctr_pred         │       │
  │  (B,1,H,W)    (B,4,H,W)    (B,1,H,W)        │       │
  │       │             │            │           │       │
  │       └──────┬──────┘            │           │       │
  │              ▼                   │           │       │
  │      Detection Loss ← CVAT GT   │           │       │
  │      (Focal+L1+BCE)             │           │       │
  │              [仅用于 loss]       │           │       │
  │                                 │           │       │
  │  ┌──────────────────────────────┼───────┐   │       │
  │  │  [训练+推理 — 特征注入]       │       │   │       │
  │  │                               │       │   │       │
  │  │  cls_f (128ch) + reg_f (128ch)│       │   │       │
  │  │       │                       │       │   │       │
  │  │       ▼                       ▼       │   │       │
  │  │  cat → (256ch)  ctr_attn ←───┘       │   │       │
  │  │       │              │                │   │       │
  │  │       ▼              ▼                │   │       │
  │  │  ×(1.0+ctr_attn)  空间门控           │   │       │
  │  │       │                               │   │       │
  │  │       ▼                               │   │       │
  │  │  1×1 Conv 256→512                    │   │       │
  │  │  → FCOS inject tokens (300)          │   │       │
  │  └──────────────────────────────────────┘   │       │
  └──────────────────────┬───────────────────────┘       │
                         │                               │
  ┌──────────────────────┼───────────────────────────┐   │
  │              Mask Decoder                         │   │
  │                                                   │   │
  │  ┌─────────────────────────────────────────────┐  │   │
  │  │  [训练 only — 用于 loss]                    │  │   │
  │  │  P2,P3,P4 → reduce → fuse → ×8 upsample    │  │   │
  │  │       → predict → pred_mask (480×640)       │  │   │
  │  │       → L1 Loss ← SAM2 GT mask              │  │   │
  │  │               [仅用于 loss]                  │  │   │
  │  └─────────────────────────────────────────────┘  │   │
  │                                                   │   │
  │  ┌─────────────────────────────────────────────┐  │   │
  │  │  [训练+推理 — 特征注入]                      │  │   │
  │  │                                             │  │   │
  │  │  P2,P3,P4 → reduce → fuse_43 → fuse_432    │  │   │
  │  │       → f432 (32ch, 60×80)   ← 注入点       │  │   │
  │  │       → 1×1 Conv 32→512                    │  │   │
  │  │       → adaptive_avg_pool → (15,20)         │  │   │
  │  │       → Mask inject tokens (300)            │  │   │
  │  └─────────────────────────────────────────────┘  │   │
  └──────────────────────┬────────────────────────────┘   │
                         │                               │
  ┌──────────────────────┼───────────────────────────┐   │
  │          Detection-Feature Fusion                │   │
  │          (空间注意力增强 F4)        [训练+推理]    │   │
  │  P2,P3,P4 → attn map → ×F4 → enhanced_f4        │   │
  │                   (512ch, 15×20)                 │   │
  └──────────────────────┬────────────────────────────┘   │
                         │                               │
                         ▼                               ▼
                  enhanced_f4                          f4
                  (512, 15×20)                   (512, 15×20)
                         │                               │
                         └───────────────┬───────────────┘
                                         │
                        ┌────────────────▼────────────────┐
                        │    1×1 Conv: 512 → dim_model     │
                        │    flatten: (H×W, B, 512)       │  [训练+推理]
                        │    → image tokens ×300/camera    │
                        └────────────────┬────────────────┘
                                         │
  ┌──────────────────────────────────────┼──────────────────────────┐
  │                        Transformer Encoder                       │
  │                                                                 │
  │  Token 序列 (V6 全注入, top+wrist 双视角):                       │
  │  ┌──────┬──────┬───────────┬───────────┬───────────┬──────────┐ │
  │  │latent│robot │top_img    │top_fcos   │top_mask   │wrist_img │ │
  │  │  ×1  │_state│  ×300     │  ×300     │  ×300     │  ×300    │ │
  │  │      │  ×1  │           │  (创新2)   │  (创新3)   │          │ │
  │  └──────┴──────┴───────────┴───────────┴───────────┴──────────┘ │
  │  + 1D/2D sinusoidal positional embeddings                       │
  └──────────────────────┬───────────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────┐
  │            Transformer Decoder                    │
  │  learned queries ×100 → cross-attn → 逐层解码    │
  └──────────────────────┬───────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────┐
  │              Action Head (Linear → 6)             │
  │              → actions (B, 100, 6)                │
  └──────────────────────┬───────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Action L1 Loss    │
              │   (vs action GT)    │
              └─────────────────────┘
```

**图中「训练 only」路径**：FCOS 预测层（cls_logits / reg_pred / ctr_pred → Detection Loss）和 Mask 预测层（pred_mask → Mask Loss）仅在训练时运行，其梯度反传塑造 Backbone 和 FPN。

**图中「训练+推理」路径**：FCOS tower 中间特征注入和 Mask Decoder 中间特征 (f432) 注入在训练和推理时都运行——检测/分割知识**显式**进入 Encoder，不是仅靠梯度雕刻。

### 2.2 推理架构

```
                          ┌──────────────────────┐
                          │      输  入  层       │
                          │  top / wrist RGB      │
                          │  robot state          │
                          │  latent = 全零向量    │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │  Shared ResNet18      │  ✅ 运行
                          │  → F2, F3, F4        │
                          └──────────┬───────────┘
                                     │
                   ┌─────────────────┴───────────────┐
                   │                                 │
          ┌────────▼────────┐              ┌─────────▼────────┐
          │  top 视角        │              │  wrist 视角       │
          │  → FPN           │  ✅ 运行     │  → F4 (原始)      │  ✅ 运行
          │  → P2, P3, P4    │              │                   │
          └──┬──────┬──────┬─┘              └─────────┬─────────┘
             │      │      │                          │
             │      │      │                          │
    ┌────────▼──────▼──────▼─────────┐                │
    │        FCOS Head               │                │
    │  cls_tower ✅  reg_tower ✅    │  ← tower 运行  │
    │  ctr_tower ✅                  │                │
    │                                │                │
    │  cls_logits ❌  reg_pred ❌    │  ← 预测层丢弃  │
    │  ctr_pred → 用于门控 ✅        │                │
    │                                │                │
    │  → FCOS inject tokens (300)    │  ✅ 创新2      │
    └───────────────┬────────────────┘                │
                    │                                 │
    ┌───────────────▼────────────────┐                │
    │       Mask Decoder             │                │
    │  reduce → fuse_43 → fuse_432   │  ← 中间层运行  │
    │  → f432 (32ch, 60×80)          │  ✅             │
    │                                │                │
    │  ×8 upsample → predict  ❌     │  ← 预测层丢弃  │
    │                                │                │
    │  → Mask inject tokens (300)    │  ✅ 创新3      │
    └───────────────┬────────────────┘                │
                    │                                 │
    ┌───────────────▼────────────────┐                │
    │  Detection-Feature Fusion      │  ✅ 运行       │
    │  → enhanced_f4 (512, 15×20)    │                │
    └───────────────┬────────────────┘                │
                    │                                 │
                    ▼                                 ▼
             enhanced_f4                             f4
                    │                                 │
                    └─────────────┬───────────────────┘
                                  │
                   ┌──────────────▼──────────────┐
                   │  1×1 Conv → dim_model        │
                   │  flatten → image tokens      │
                   └──────────────┬──────────────┘
                                  │
   ┌──────────────────────────────┼──────────────────────────────┐
   │                   Transformer Encoder                        │
   │  Token 序列:                                                 │
   │  [latent|state|top_img×300|top_fcos×300|top_mask×300         │
   │   |wrist_img×300]                                            │
   └──────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
   ┌──────────────────────────────────────────────┐
   │  Transformer Decoder → Action Head            │
   │  → actions (1, 100, 6)  关节目标位置序列       │
   └──────────────────────────────────────────────┘

图例:  ✅ 推理时运行    ❌ 推理时丢弃 (仅训练时用于 loss)
```

**推理时各版本的实际路径：**

| 版本           | top 视角推理路径                                       | Encoder token 组成         |
| -------------- | ------------------------------------------------------ | -------------------------- |
| V1 (标准ACT)   | `Backbone → F4`                                     | latent + state + img×600  |
| V2 (论文版)    | `Backbone → FPN → Fusion → enhanced_f4`           | latent + state + img×600  |
| V3 (论文+Mask) | 同 V2（推理路径相同，Backbone 权重被 Mask 梯度雕刻过） | 同 V2                      |
| V4 (+FCOS注入) | V2 +`FCOS tower → inject tokens`                    | V2 + fcos×300             |
| V5 (+Mask注入) | V3 +`Mask f432 → inject tokens`                     | V3 + mask×300             |
| V6 (全注入)    | V3 + FCOS tower + Mask f432 → inject tokens           | V3 + fcos×300 + mask×300 |

**关键理解**：V2 和 V3 推理路径完全相同——区别仅在训练时 V3 多了 Mask Loss 对 Backbone 的梯度雕刻。V4/V5/V6 则在此基础上增加了推理时运行的额外特征路径。

### 2.3 文件结构

```
src/lerobot/policies/act_det/
├── __init__.py                     # 导出
├── configuration_act_det.py        # 所有配置参数（~140行）
├── modeling_act_det.py             # ACTDetPolicy + ACTDetModel（~650行）
├── processor_act_det.py            # 复用 ACT 处理器
├── label_loader.py                 # CVAT XML 标注加载
├── mask_loader.py                  # NPZ mask 加载（LRU 缓存）
└── detection/
    ├── __init__.py
    ├── fpn.py                      # Feature Pyramid Network
    ├── fcos.py                     # FCOS 检测头 + 特征注入 + Focal/L1/BCE 损失
    ├── fusion.py                   # 检测-动作特征融合模块
    ├── mask_decoder.py             # Mask Decoder + 特征注入（FPN → mask）
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

## 4. 版本切换

通过 `use_detection`、`use_mask_guidance`、`fcos_feature_inject`、`mask_feature_inject` 四个参数实现多版本切换：

| 版本                    | 策略类型                | 配置                                                            | 检测信息路径         | Mask 信息路径        |
| ----------------------- | ----------------------- | --------------------------------------------------------------- | -------------------- | -------------------- |
| **V1: 标准 ACT**  | `policy.type=act`     | 原始 ACT，无 FPN/Fusion/检测/Mask                               | —                   | —                   |
| **V2: 论文版**    | `policy.type=act_det` | `use_detection=true`, `use_mask_guidance=false`             | 梯度反传             | —                   |
| **V3: 论文+Mask** | `policy.type=act_det` | `use_detection=true`, `use_mask_guidance=true`              | 梯度反传             | 梯度反传             |
| **V4: +FCOS注入** | `policy.type=act_det` | V2 +`fcos_feature_inject=true`                                | **显式 token** | —                   |
| **V5: +Mask注入** | `policy.type=act_det` | V3 +`mask_feature_inject=true`                                | 梯度反传             | **显式 token** |
| **V6: 全注入**    | `policy.type=act_det` | V3 +`fcos_feature_inject=true` + `mask_feature_inject=true` | **显式 token** | **显式 token** |

**推理架构分类：**

- V1 → `Backbone(layer4) → Transformer`
- V2/V3 → `Backbone(layer2/3/4) → FPN → Fusion → Transformer`（检测/mask 信息仅通过梯度雕刻间接进入）
- V4/V5/V6 → `Backbone(layer2/3/4) → FPN → Fusion → Transformer` + **额外注入 token**（检测/mask 信息显式进入 Encoder）

**关键对比：**

- V2 vs V4：回答"显式检测特征比纯梯度雕刻好多少？"
- V3 vs V5：回答"显式 Mask 特征有额外收益吗？"
- V3 vs V6：回答"双注入的上限在哪？"

---

## 5. 操作指南

### 5.1 环境要求

| 环境    | 配置                                       |
| ------- | ------------------------------------------ |
| OS      | Ubuntu 22.04 LTS                           |
| Python  | ≥ 3.12                                    |
| GPU     | NVIDIA RTX 4090（16GB 显存）               |
| PyTorch | ≥ 2.2.1                                   |
| LeRobot | v3.0+                                      |
| SAM 2   | segment-anything-2（仅 mask 生成脚本需要） |

### 5.2 安装

```bash
# 1. 安装 LeRobot 及其依赖
cd lerobot-main
pip install -e .

# 2. 安装 SAM 2（仅在 Ubuntu 上，仅 mask 生成阶段）
pip install segment-anything-2 opencv-python scipy
```

### 5.3 数据集准备

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

### 5.4 训练

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

### 5.5 评估

```bash
lerobot-eval \
    --policy.type=act_det \
    --policy.pretrained_path=/path/to/checkpoint \
    --dataset.repo_id=your_dataset \
    --eval.episodes=30
```

### 5.6 消融实验

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

### 5.7 关键配置参数速查

| 参数               | 默认值                | 说明                   |
| ------------------ | --------------------- | ---------------------- |
| `chunk_size`     | 100                   | 一次预测的动作序列长度 |
| `n_action_steps` | 100                   | 实际执行的动作步数     |
| `dim_model`      | 512                   | Transformer 隐藏维度   |
| `latent_dim`     | 32                    | VAE 潜在空间维度       |
| `use_vae`        | true                  | CVAE 开关              |
| `kl_weight`      | 10.0                  | KL 散度权重            |
| `use_detection`  | true                  | 检测分支开关           |
| `det_weight`     | 10.0                  | 检测损失权重           |
| `det_cameras`    | top:true, wrist:false | 检测每视角独立开关     |

| `fcos_num_classes`        | 1        | 检测类别数（cup）      |
| `focal_alpha`             | 0.25     | Focal Loss α          |
| `focal_gamma`             | 2.0      | Focal Loss γ          |
| `use_mask_guidance`       | true     | Mask 引导开关          |
| `mask_weight`             | 1.0      | Mask 损失权重          |
| `mask_cameras`            | top:true, wrist:false | Mask 每视角独立开关 |

| **`fcos_feature_inject`** | **false** | **FCOS 特征显式注入** (创新 2) |
| **`fcos_inject_levels`**  | **["p4"]** | **注入的 FPN 层级**（可选 p2/p3/p4） |
| **`mask_feature_inject`** | **false** | **Mask 特征显式注入** (创新 3) |
| **`mask_inject_pool_size`** | **(15, 20)** | **Mask 注入降采样分辨率** |

| `aug_enable`              | true     | 数据增强开关           |
| `aug_probability`         | 0.9      | 每方法独立应用概率     |
| `aug_color_jitter_enable` | true     | 色彩抖动               |
| `aug_noise_enable`        | true     | 高斯噪声               |
| `aug_occlusion_enable`    | true     | 随机遮挡               |
| `optimizer_lr`            | 1e-5     | 学习率                 |
| `optimizer_lr_backbone`   | 1e-5     | Backbone 学习率        |
| `vision_backbone`         | resnet18 | 视觉骨干网络           |

### 5.8 模型参数量

| 版本               | 参数量       |
| ------------------ | ------------ |
| V1: 标准 ACT       | ~52M         |
| V2: 论文版（检测） | ~58M         |
| V3: 论文+Mask      | ~58M + ~20K  |
| V4: +FCOS注入      | ~58M + ~131K |
| V5: +Mask注入      | ~58M + ~36K  |
| V6: 全注入         | ~58M + ~167K |

---

## 6. 快速使用说明

### 6.1 多版本一键命令

```bash
# V1: 标准ACT
lerobot-train --policy.type=act

# V2: 论文版（检测，Top视角检测 + 无 Mask，纯梯度反传）
lerobot-train --policy.type=act_det --policy.use_mask_guidance=false

# V3: 检测+Mask（Top视角检测 + Top视角Mask，纯梯度反传，默认）
lerobot-train --policy.type=act_det --policy.use_mask_guidance=true

# V4: 论文版 + FCOS特征显式注入（创新2）
lerobot-train --policy.type=act_det \
    --policy.use_mask_guidance=false \
    --policy.fcos_feature_inject=true \
    --policy.fcos_inject_levels '["p4"]'

# V5: 检测+Mask + Mask特征显式注入（创新3）
lerobot-train --policy.type=act_det \
    --policy.use_mask_guidance=true \
    --policy.mask_feature_inject=true \
    --policy.mask_inject_pool_size '(15, 20)'

# V6: 全注入（创新1+2+3 全开）
lerobot-train --policy.type=act_det \
    --policy.use_mask_guidance=true \
    --policy.fcos_feature_inject=true \
    --policy.mask_feature_inject=true

# 双视角全开（Top+Wrist 检测 + Top+Wrist Mask）
lerobot-train --policy.type=act_det \
    --policy.use_mask_guidance=true \
    --policy.det_cameras '{"observation.images.top":{"enable":true},"observation.images.wrist":{"enable":true}}' \
    --policy.mask_cameras '{"observation.images.top":{"enable":true},"observation.images.wrist":{"enable":true}}'
```

### 6.2 在 Ubuntu 运行前需要做的

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
