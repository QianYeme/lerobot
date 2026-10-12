# Experiment Decision Contract

> Revision: r0  
> 本文件由用户与 ChatGPT 冻结，Codex 只读。

# P0 — Cup Prototype Calibration

## 1. Identity

- **Experiment ID**: `P0_CUP_PROTOTYPE_CALIBRATION`
- **Track**: `POSITION`
- **Stage**: `OBJECT_REFERENCE_CALIBRATION`
- **Mode**: `DIAGNOSIS + ENGINEERING`
- **Upstream**: `M0_OC_ACT_ARCHITECTURE_BUILD = PASS`
- **Formal Policy Training**: 禁止
- **Dev / Final Test Use**: 禁止
- **Real-Robot Execution**: 禁止

本轮唯一目标：

> 使用 Train-only 或独立 calibration reference 图像，为半满透明塑料杯建立正式、冻结、可复现的 DINOv2-S Cup Prototype Bank，并验证其 similarity map 与 5 个 object tokens 是否确实捕获目标杯而非背景 shortcut。

本轮结束后才能进入正式 ACT / OC-ACT 训练。

---

# 2. Scientific Question

> **RQ:** 在不训练 detector、不过度标注完整数据集的前提下，少量 reference images + Frozen DINOv2-S 是否能够产生稳定、空间正确、跨 Train 场景可重复的透明杯 object similarity representation？

本轮不是 policy-performance 实验。

---

# 3. Data Boundary

## 3.1 Allowed Data

只允许：

```text
Train split
```

或：

```text
独立 Reference Calibration Set
```

若使用 Train split，必须从 D1 冻结的：

```text
split_manifest_v1.json
```

中明确筛选 `split=train`。

## 3.2 Forbidden Data

禁止读取或使用：

```text
Dev
Final Test
```

用于：

- reference 选择；
- prototype 构建；
- prototype 数量选择；
- threshold / temperature 选择；
- similarity map 调整；
- qualitative tuning。

若发现误用：

```text
TEST_CONTAMINATED
```

本轮立即停止。

---

# 4. DINO Contract

固定使用：

```text
dinov2_vits14
```

要求：

- 与 M0 完全相同的 pretrained checkpoint / model identifier；
- Frozen；
- resize / normalization 与 M0 一致；
- 不 fine-tune；
- 不使用 action loss 更新。

---

# 5. Reference Image Selection

## 5.1 Recommended Count

第一版建议：

```text
8 reference images
```

允许范围：

```text
5–10
```

但最终数量必须在 prototype 构建前记录并冻结。

## 5.2 Coverage Goal

reference images 应尽量覆盖 Train 中：

- 横向不同位置；
- 纵向不同位置；
- 轻微不同反光状态；
- 正常杯体外观变化。

不要求覆盖全部 36 Train episodes。

## 5.3 Selection Rule

优先使用：

> 分层、可复现、非结果导向的 reference selection。

例如：

- 从 Train 的不同 spatial strata 中取样；
- 避免全部来自 nominal center；
- 避免人为挑选“DINO 看起来最好”的帧。

必须保存：

```text
reference_manifest.json
```

至少记录：

- episode_index；
- frame_index；
- source split；
- selection reason；
- image path / dataset key。

---

# 6. Minimal Cup Annotation

本轮允许少量人工标注，但目标是：

> 不再回到全数据集 dense box supervision。

第一版允许：

```text
coarse box
```

或：

```text
few point / patch selection
```

推荐优先使用：

> 粗 box + DINO patch-level filtering。

不要求 pixel-perfect mask。

## 6.1 Annotation Scope

只标注 reference images。

禁止：

- 标注全部 Train frames；
- 标注 Dev / Final Test；
- 训练 FCOS / detector；
- 使用旧 detector 自动生成正式标签后再声称 detector-free。

---

# 7. Transparent Cup Special Handling

透明杯存在：

- 背景透过杯体；
- 反光；
- 水面线；
- 杯沿；
- 杯壁边缘；
- 杯底轮廓。

因此禁止简单执行：

```text
box 内所有 patch 全部平均
```

而不检查污染。

必须至少可视化 box 内 patch similarity / feature grouping，确认 prototype 没有主要编码：

- 桌面纹理；
- 角标；
- 固定背景；
- gripper；
- 其他高对比 shortcut。

---

# 8. Prototype Bank Construction

## 8.1 Source Features

从每个 reference image：

```text
Frozen DINOv2-S
→ patch features
```

结合人工 cup region 选择 cup-related patches。

## 8.2 First-Version Bank

目标 bank：

```text
4–8 prototypes
```

默认优先尝试：

```text
4 prototypes
```

除非证据显示单纯 4 个无法覆盖 reference variation。

Prototype 可通过以下可复现方式之一生成：

- clustering；
- medoid；
- robust mean of coherent cup patches；
- region-specific aggregation。

禁止手工直接编辑 feature 数值。

## 8.3 Frozen Artifact

最终输出：

```text
cup_prototype_bank_v1.pt
```

必须为 frozen artifact，并记录：

- DINO identifier；
- feature dim；
- prototype count；
- source reference IDs；
- construction algorithm；
- creation script version / git commit；
- checksum/hash。

---

# 9. Similarity Rule

与 M0 保持一致：

```text
cosine similarity
```

对当前 patch：

```text
z_i
```

与 bank：

```text
p_j
```

第一版默认：

```text
s_i = max_j cosine(z_i, p_j)
```

允许额外输出 soft aggregation 供诊断，但正式规则必须冻结为一个版本。

本轮不允许根据 Dev/Test 调 similarity rule。

---

# 10. No Hard Threshold as Primary Representation

正式 OC-ACT 第一版仍使用：

```text
soft similarity
```

禁止把目标识别完全变成：

```text
similarity > threshold → binary mask
```

理由：

- 透明杯边界不稳定；
- hard threshold 容易对反光敏感；
- 当前架构需要 soft pooling。

阈值只允许用于 visualization / diagnostic，不作为唯一 object representation。

---

# 11. Evaluation Set Within Train

为避免 reference self-evaluation，Train 内部临时区分：

```text
Reference subset
Calibration-check subset
```

`Calibration-check subset` 仍属于 Train，不改变正式 Train/Dev/Test split。

要求：

> 至少在未被用于 prototype construction 的 Train frames 上检查 similarity map。

本轮不得查看 Dev / Final Test。

---

# 12. Mandatory Visual Diagnostics

必须生成多张 Top-image overlay：

```text
original top RGB
+
DINO patch similarity heatmap
+
soft center
+
2×2 local regions
```

至少覆盖：

- reference frames；
- unseen-in-prototype Train frames；
- 横向左/中/右；
- 纵向近/中/远；
- 不同轻微反光情况。

---

# 13. Spatial Correctness Metrics

本轮不要求 detector mAP。

优先使用适合 prototype calibration 的简单指标。

如果 reference / check frames 有 coarse cup box：

至少报告：

## 13.1 Soft Center Error

similarity-weighted soft center：

```text
(x_soft, y_soft)
```

相对于 coarse cup center 的误差：

```text
pixels
normalized image coordinates
```

## 13.2 Similarity Concentration

比较：

```text
inside cup region similarity mass
vs
outside region similarity mass
```

## 13.3 Peak-in-Region Rate

最高 similarity patch 是否落在 cup region 或其合理邻域。

## 13.4 Cross-Frame Stability

对不同位置 / 反光状态，报告：

- center error distribution；
- peak consistency；
- similarity mass stability。

---

# 14. Background Shortcut Tests

必须至少执行：

## P0-GS1 — Corner Marker Check

检查高 similarity 是否系统性落在：

```text
低反光角标
```

而不是杯体。

## P0-GS2 — Table Background Check

检查 similarity 是否主要锁定固定桌纹 / 背景。

## P0-GS3 — Position Shift Check

杯子位置变化时：

> soft center 应随杯位置变化，而不是固定停在 nominal P0。

## P0-GS4 — Reference Memorization Check

未用于 prototype construction 的 Train frames 上仍应有目标响应。

---

# 15. Object Token Validation

用正式 candidate prototype bank 生成：

```text
T0 = global
T1–T4 = local
```

检查：

- shape = `[B,5,387]`（若 DINO-S=384 且附加 x/y/confidence）；
- x/y 随目标位置变化；
- confidence 有合理动态范围；
- local token 不全部退化为完全相同向量；
- global token 非 NaN / Inf；
- 空局部区域有稳定 fallback 规则。

---

# 16. Prototype Count Selection

本轮允许在 Train-only 内比较少量候选，例如：

```text
1
4
8
```

但必须遵循：

> 只以 Train calibration-check 的 object-localization diagnostic 选择，不使用 policy loss，不使用 Dev/Test。

推荐优先顺序：

```text
4
→ 若明显不足再 8
```

不为了追求极小 metric 改进无限扩大 bank。

---

# 17. PASS Criteria

P0 不设一个武断的单数字阈值。

### PASS

需要同时满足：

1. similarity heatmap 在绝大多数 Train check frames 上主要响应杯体/稳定杯结构；
2. soft center 会随杯位置正确移动；
3. 不明显锁定角标、桌面或 nominal P0；
4. 未参与 prototype construction 的 Train frames 仍能响应；
5. 5 tokens 数值稳定；
6. prototype artifact provenance 完整；
7. Dev / Final Test 完全未使用。

### PARTIAL

例如：

- 杯沿 / 水面响应稳定，但杯身中心弱；
- 横向位置正确，纵向略不稳定；
- 少数强反光 frame 失效；
- 4 prototypes 不够，但 8 prototypes 明显改善。

PARTIAL 时不得自动进入正式训练，由用户 + ChatGPT 决定。

### FAIL

例如：

- similarity 主要锁定桌面；
- 主要锁定角标；
- soft center 不随杯移动；
- 只在 reference 自身有效；
- prototype bank 对多数 Train check frames 无目标响应。

---

# 18. Required Outputs

建议目录：

```text
experiments/position/p0_prototype_calibration/index1/
├── decision.md
└── result/
    ├── summary.md
    ├── validation.json
    ├── metrics.json
    ├── reference_manifest.json
    ├── prototype_provenance.json
    ├── cup_prototype_bank_v1.pt
    ├── figures/
    │   ├── reference_similarity_overlays/
    │   ├── train_check_similarity_overlays/
    │   ├── soft_center_error.*
    │   ├── similarity_concentration.*
    │   └── prototype_feature_summary.*
    └── proposal.md        # optional
```

---

# 19. Codex MAY

- 从 Train split 读取图像；
- 提取 DINO features；
- 创建少量 reference annotation helper；
- 保存 coarse boxes / points；
- 聚类 / 聚合 prototype；
- 比较 1/4/8 prototype；
- 生成 similarity overlays；
- 生成 object tokens；
- 创建 proposal.md。

---

# 20. Codex MUST

- 严格 Train-only；
- 保存 reference manifest；
- 保存 annotation provenance；
- 保存 prototype construction algorithm；
- 检查 background shortcut；
- 检查 unseen Train frames；
- 保持 DINO frozen；
- 不使用 policy performance 选择 prototype；
- 完成后停止。

---

# 21. Codex MUST NOT

- 修改本 decision.md；
- 读取 Dev / Final Test；
- 训练 ACT / OC-ACT；
- fine-tune DINO；
- 训练 detector；
- 使用 FCOS 正式生成 reference；
- 使用 SAM 作为新的正式研究模块；
- 修改 OC-ACT architecture；
- 启动 real robot；
- 自动进入正式训练。

---

# 22. Completion Rule

只有 P0 验收为：

```text
PASS
```

才允许创建下一阶段：

```text
T0 — Formal ACT Baseline Training
T1 — Formal OC-ACT Training
```

P0 结束时必须停止，等待用户 + ChatGPT 验收。
