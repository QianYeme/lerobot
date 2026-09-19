# M-VF-ACT 实验计划

> 力控抓取半满透明塑料杯 — 横向对比实验方案
>
> 创建日期：2026-07-17

---

## 1. 当前状态总览

| 模块 | 代码 | 文档 | 测试 |
|------|------|------|------|
| 标准 ACT | ✅ 已有 | ✅ | 待跑 |
| 论文版（FPN+FCOS+融合） | ✅ `act_det` | ✅ `REAMDE_MVF_ACT.md` | 待跑 |
| 创新1: Mask引导感知 | ✅ `act_det` + mask_decoder | ✅ | 待跑 |
| 创新2: FCOS特征注入 | ✅ `act_det` + fcos.py | ✅ | 待跑 |
| 创新3: Mask特征注入 | ✅ `act_det` + mask_decoder.py | ✅ | 待跑 |
| 创新4: Visual-Force Fusion | 📋 规划中 | — | 待主实验后定 |
| 创新5: Hybrid Action Head | 📋 规划中 | — | 待定 |
| 创新6: Temporal Modeling | 📋 规划中 | — | 待定 |
| SAM 2 Mask 生成脚本 | ✅ `scripts/generate_sam2_masks.py` | — | 待跑 |

---

## 2. 数据集

### 2.1 数据集划分

| 数据集 | state维度 | 内容 | 总episode | 训练episode | 测试episode |
|--------|----------|------|-----------|-------------|-------------|
| **A（无力感）** | 6 | kind1(固定50) + kind2(随机40) | 90 | 63 (70%) | 27 (30%) |
| **B（有力感）** | 9 | kind1(固定50) + kind2(随机40) | 90 | 63 (70%) | 27 (30%) |

**测试方式**：测试集的30个episode使用训练期间从未出现过的随机初始位置，验证跨位置泛化能力。

### 2.2 observation.state 维度

| 数据集 | 维度 | 组成 |
|--------|------|------|
| A | 6 | shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos |
| B | 9 | 上述6维 + gripper.load, gripper.curr, master_gripper.pos |

---

## 3. 训练参数

| 参数 | 值 | 依据 |
|------|-----|------|
| `steps` | 120,000 | 63ep×718帧/batch=8 → ～21 epochs，与论文一致 |
| `batch_size` | 8 | 16GB VRAM 边界 |
| `chunk_size` | 100 | 论文默认，覆盖 ~3.3s |
| `n_action_steps` | 100 | 与chunk_size一致 |
| `dim_model` | 512 | 论文默认 |
| `latent_dim` | 32 | 论文默认 |
| `use_vae` | true | CVAE 框架 |
| `kl_weight` | 10.0 | 论文默认 |
| `det_weight` | 10.0 | 检测损失权重 |
| `mask_weight` | 1.0 | Mask 损失权重 |
| `optimizer_lr` | 1e-5 | 论文默认 |
| `optimizer_lr_backbone` | 1e-5 | Backbone 学习率 |
| `vision_backbone` | resnet18 | ImageNet 预训练 |
| `aug_enable` | true | 数据增强 |
| `aug_probability` | 0.9 | 每方法独立概率 |
| `focal_alpha` | 0.25 | Focal Loss |
| `focal_gamma` | 2.0 | Focal Loss |
| `fcos_num_classes` | 1 | cup 单类 |
| `fcos_feature_inject` | false | FCOS特征注入开关（创新2），E7/E9 开启 |
| `fcos_inject_levels` | ["p4"] | 注入的FPN层级，可选 p2/p3/p4 |
| `mask_feature_inject` | false | Mask特征注入开关（创新3），E8/E9 开启 |
| `mask_inject_pool_size` | (15, 20) | Mask注入降采样分辨率 |

---

## 4. 核心实验矩阵（9组）

| ID | 数据集 | 模型 | 命令 |
|----|--------|------|------|
| **E1** | A (无力感) | 标准 ACT (V1) | `lerobot-train --policy.type=act --dataset.repo_id=dataset_A` |
| **E2** | A (无力感) | 论文版 检测 (V2) | `lerobot-train --policy.type=act_det --policy.use_mask_guidance=false --policy.annotation_dir=...` |
| **E3** | A (无力感) | 检测+Mask (V3) | `lerobot-train --policy.type=act_det --policy.use_mask_guidance=true --policy.annotation_dir=...` |
| **E4** | B (有力感) | 标准 ACT (V1) | `lerobot-train --policy.type=act --dataset.repo_id=dataset_B` |
| **E5** | B (有力感) | 论文版 检测 (V2) | `lerobot-train --policy.type=act_det --policy.use_mask_guidance=false --policy.annotation_dir=...` |
| **E6** | B (有力感) | 检测+Mask (V3) | `lerobot-train --policy.type=act_det --policy.use_mask_guidance=true --policy.annotation_dir=...` |
| **E7** | B (有力感) | +FCOS注入 (V4) | `lerobot-train --policy.type=act_det --policy.use_mask_guidance=false --policy.fcos_feature_inject=true --policy.annotation_dir=...` |
| **E8** | B (有力感) | +Mask注入 (V5) | `lerobot-train --policy.type=act_det --policy.use_mask_guidance=true --policy.mask_feature_inject=true --policy.annotation_dir=...` |
| **E9** | B (有力感) | 全注入 (V6) | `lerobot-train --policy.type=act_det --policy.use_mask_guidance=true --policy.fcos_feature_inject=true --policy.mask_feature_inject=true --policy.annotation_dir=...` |

**对比维度：**

| 对比 | 实验对 | 验证什么 |
|------|--------|----------|
| 检测 vs 基线 | E2-E1, E5-E4 | FCOS检测对泛化能力的贡献 |
| Mask vs 检测 | E3-E2, E6-E5 | 像素级Mask监督（梯度雕刻）的增量贡献 |
| **FCOS注入 vs 纯梯度** | **E7-E5** | **显式检测特征比纯梯度雕刻好多少？** |
| **Mask注入 vs 纯梯度** | **E8-E6** | **显式Mask特征比纯梯度雕刻好多少？** |
| **双注入上限** | **E9-E6** | **双注入的上限在哪？** |
| 力感数据 vs 无感数据 | E4-E1, E5-E2, E6-E3 | 力感信号本身的基线提升 |

---

## 5. 子实验矩阵（按需）

### 5.1 数据增强消融（在最佳基础模型上）

| 子实验 | 配置 |
|--------|------|
| 全增强（默认） | 色彩抖动 + 噪声 + 遮挡，prob=0.9 |
| 仅噪声 | `aug_color_jitter_enable: false`, `aug_occlusion_enable: false` |
| 仅色彩抖动 | `aug_noise_enable: false`, `aug_occlusion_enable: false` |
| 无增强 | `aug_enable: false` |

### 5.2 检测视角消融（在 E5/E6 上）

| 子实验 | 配置 |
|--------|------|
| Top检测（默认） | `det_cameras.top.enable: true`, `det_cameras.wrist.enable: false` |
| 双视角检测 | `det_cameras.top.enable: true`, `det_cameras.wrist.enable: true` |

### 5.3 Mask视角消融（在 E6 上）

| 子实验 | 配置 |
|--------|------|
| Top Mask（默认） | `mask_cameras.top.enable: true`, `mask_cameras.wrist.enable: false` |
| 双视角Mask | `mask_cameras.top.enable: true`, `mask_cameras.wrist.enable: true` |

### 5.4 Mask权重消融（在 E6 上）

| 子实验 | 配置 |
|--------|------|
| mask_weight=1.0（默认） | 标准 |
| mask_weight=5.0 | 加重 |
| mask_weight=0.5 | 减轻 |

### 5.5 FCOS 注入层级消融（在 E7 上）

| 子实验 | 配置 | Token 增量 |
|--------|------|-----------|
| 仅 P4（默认） | `fcos_inject_levels: ["p4"]` | +300 |
| P3+P4 | `fcos_inject_levels: ["p3", "p4"]` | +1500 |
| P2+P3+P4 | `fcos_inject_levels: ["p2", "p3", "p4"]` | +6300 |

### 5.6 Mask 注入分辨率消融（在 E8 上）

| 子实验 | 配置 | Token 增量 |
|--------|------|-----------|
| pool→15×20（默认） | `mask_inject_pool_size: (15, 20)` | +300 |
| pool→8×10 | `mask_inject_pool_size: (8, 10)` | +80 |
| pool→4×5 | `mask_inject_pool_size: (4, 5)` | +20 |

---

## 6. 评价指标

### 6.1 主要指标：任务成功率

和论文一致，分解为四阶段：

| 阶段 | 度量 | 依赖前阶段 |
|------|------|-----------|
| 抓取杯子 | 成功抓取/总测试 | 独立 |
| 移动杯子 | 成功移向目标/总测试 | → 抓取成功 |
| 放下杯子 | 成功放置/总测试 | → 移动成功 |
| 整体成功率 | 完整完成/总测试 | → 全阶段通过 |

### 6.2 辅助指标

| 指标 | 记录方式 | 用途 |
|------|----------|------|
| Action L1 Loss (val) | 每个 eval step | 收敛验证 |
| Detection Loss (val) | 每个 eval step | 检测分支收敛 |
| Mask Loss (val) | 每个 eval step | Mask Decoder 收敛 |
| 力感信号变化 | 力感数据集 | 接触时刻检测 |
| Encoder token 总数 | 模型初始化时 | 确认注入参数正确，避免 OOM |
| 注入 token 的 attention 权重 | WandB histogram（可选） | 分析 Transformer 是否实际使用了注入 token |

---

## 7. 执行顺序

```
Phase A: 环境准备（1天）
├── 1. Ubuntu安装 LeRobot + 依赖
├── 2. 确认数据集A、B在LeRobot格式下可加载
├── 3. 运行 SAM 2 mask 生成脚本（对数据集A和B各跑一遍）
├── 4. 验证 annotations/ 目录结构完整
└── 5. 小批量测试（steps=1000）验证所有版本可训练

Phase B: 核心实验 9 组（每组 ~8h × 9 = ~72h）
├── B1: 数据集A → E1, E2, E3  (验证检测+Mask在无力感数据上的基础效果)
├── B2: 数据集B → E4, E5, E6  (验证检测+Mask在有力感数据上的基础效果)
└── B3: 数据集B → E7, E8, E9  (验证特征显式注入 vs 纯梯度雕刻的增量收益)

Phase C: 结果分析（1天）
├── 9组横向对比表（成功率矩阵）
├── 绘制训练/验证损失曲线（action/detection/mask）
├── 分析失败模式（抓取失败 vs 移动失败 vs 放下失败）
├── 核心对比: E7-E5（FCOS显式注入 vs 纯梯度雕刻）
├── 核心对比: E8-E6（Mask显式注入 vs 纯梯度雕刻）
├── 核心对比: E9-E6（双注入上限 vs 纯梯度雕刻）
└── 根据失败模式 + 注入收益决定下一步方向

Phase D: 子实验（按需，每组 ~8h）
├── 数据增强消融（§5.1）
├── 检测/Mask视角消融（§5.2, §5.3）
├── Mask权重调优（§5.4）
├── FCOS注入层级消融（§5.5）— 仅当 E7 有效时
└── Mask注入分辨率消融（§5.6）— 仅当 E8 有效时
```

---

## 8. 决策门

**Phase C 完成后，根据失败模式决定下一步：**

| 主要失败模式 | 下一步 |
|-------------|--------|
| 杯子定位不准（抓取位置偏差） | 优化 Mask Decoder / 检查 SAM 2 mask 质量 / 对比 V5 vs V3 |
| FCOS注入有效(E7>E5)但Mask注入无效(E8≈E6) | 检测信号已足够，Mask 注入可关闭，推进创新4 |
| Mask注入有效(E8>E6)但FCOS注入无效(E7≈E5) | 分割信号是关键，考虑加大 mask_weight 或在 wrist 视角也开 Mask 注入 |
| 接触判断延迟（捏晚/捏碎/滑落） | **实现创新4：Visual-Force Fusion** |
| 夹爪力度不适（不同水量不稳） | 实现创新5：Hybrid Action Head |
| 水晃动导致丢失（移动时杯不稳） | 实现创新6：Temporal Modeling |
| 所有阶段都有进步，但总成功率<50% | 检查数据规模是否需要增加 |
| 成功率满意（>60%） | 直接写论文，创新4/5/6留作未来工作 |

---

## 9. 日志与记录

### 9.1 每组实验需记录

```yaml
实验ID: E1
日期: 2026-07-xx
模型: 标准 ACT
数据集: A (无力感, 6-dim state)
关键参数:
  steps: 120000, batch: 8, chunk: 100
  fcos_feature_inject: false
  mask_feature_inject: false
训练耗时: Xh
最终val_loss: X.XX
Encoder token 总数: X
测试集30ep结果:
  抓取成功率: X/30
  移动成功率: X/30
  放下成功率: X/30
  整体成功率: X/30 (xx.x%)
失败分析:
  - 主要失败模式: xxx
  - 典型失败帧: episode_X frame_Y
```

### 9.2 WandB 看板

每个实验推送到 WandB，用 `project: "mvfact"` 统一管理：
```bash
# V1-V3 基线
lerobot-train ... --wandb.project=mvfact --wandb.run_name=E1_act_baseline_A
lerobot-train ... --wandb.project=mvfact --wandb.run_name=E2_det_A
lerobot-train ... --wandb.project=mvfact --wandb.run_name=E3_det_mask_A
lerobot-train ... --wandb.project=mvfact --wandb.run_name=E4_act_baseline_B
lerobot-train ... --wandb.project=mvfact --wandb.run_name=E5_det_B
lerobot-train ... --wandb.project=mvfact --wandb.run_name=E6_det_mask_B

# V4-V6 特征注入消融
lerobot-train ... --wandb.project=mvfact --wandb.run_name=E7_fcos_inject_B
lerobot-train ... --wandb.project=mvfact --wandb.run_name=E8_mask_inject_B
lerobot-train ... --wandb.project=mvfact --wandb.run_name=E9_full_inject_B
```

---

## 10. 下一步操作清单

- [ ] **Ubuntu环境**：确认 RTX 4090 + CUDA 11.8 可用
- [ ] **安装依赖**：`pip install -e . && pip install segment-anything-2`
- [ ] **准备标注**：确保所有 180ep 的 CVAT XML 标注文件就位
- [ ] **生成 Mask**：`python scripts/generate_sam2_masks.py --dataset_root /path/to/dataset_A`
- [ ] **生成 Mask**：`python scripts/generate_sam2_masks.py --dataset_root /path/to/dataset_B`
- [ ] **验证训练**：`lerobot-train --policy.type=act_det --training.steps=1000` 测试 V2 可训练
- [ ] **验证注入**：`lerobot-train --policy.type=act_det --policy.fcos_feature_inject=true --policy.mask_feature_inject=true --training.steps=1000` 测试 V6 可训练，确认 token 数无异常
- [ ] **启动 E1-E6**：V1-V3 基线实验（~48h）
- [ ] **启动 E7-E9**：V4-V6 特征注入消融（~24h），可与 E4-E6 并行对比

---

> **决策原则：数据驱动，先跑基线（E1-E6，V1-V3），通过 E7-E9 验证特征显式注入的增量收益，根据失败日志决定是否推进创新4/5/6。**
