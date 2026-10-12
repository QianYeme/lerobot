# Experiment Decision Contract

> Revision: r0  
> 本文件由用户与 ChatGPT 冻结，Codex 只读。

# P0.1 — Cup Spatial Pooling Fix

## 1. Identity

- **Experiment ID**: `P0_1_CUP_SPATIAL_POOLING_FIX`
- **Track**: `POSITION`
- **Stage**: `OBJECT_REFERENCE_CALIBRATION`
- **Mode**: `DIAGNOSIS`
- **Upstream**: `P0_CUP_PROTOTYPE_CALIBRATION = FAIL`
- **Formal Policy Training**: 禁止
- **Dev / Final Test Use**: 禁止
- **Real-Robot Execution**: 禁止

本轮只解决一个已确认问题：

> P0 中 prototype similarity peak 能正确落在杯子区域，但 untempered global softmax 太分散，导致 soft center 长期停留在接近固定图像中心的位置，不能跟随杯子移动。

---

# 2. Frozen Upstream Evidence

P0 已确认：

```text
selected prototype count = 4
peak in coarse cup box = 12 / 12
cup beats corner = 12 / 12
inside/outside similarity ratio mean ≈ 2.786
soft-center median error ≈ 78.28 px
soft-center P90 error ≈ 97.77 px
inside softmax mass mean ≈ 0.0508
```

因此本轮假设：

> Prototype identification itself is usable; the failed component is spatial pooling concentration.

---

# 3. Scientific Question

> **RQ:** 在保持 DINO、prototype bank、reference set 和 OC-ACT 架构不变的情况下，temperature-scaled soft pooling 是否能让 object soft center 与杯子真实位置同步移动，同时保持 detector-free、soft representation 的设计？

---

# 4. Frozen Components

本轮必须保持不变：

```text
DINO = dinov2_vits14
DINO weights = frozen
Prototype Bank = P0 selected 4 prototypes
Reference images = P0 same 8
Calibration-check images = P0 same 12
Coarse boxes = P0 same annotations
Similarity = max cosine over prototype bank
Object token count = 5
Token semantic source = DINO patch features
OC-ACT architecture = unchanged
```

禁止重新：

- 选择 reference；
- 修改 box；
- 重新构建 prototype；
- 改 prototype 数量；
- 改 DINO；
- 使用 Dev/Test。

---

# 5. Only Authorized Intervention

当前 P0：

```text
w_i = softmax(s_i)
```

本轮只允许改为：

```text
w_i(tau) = softmax(s_i / tau)
```

其中：

```text
tau > 0
```

tau 越小，权重越集中在高相似 patch。

---

# 6. Candidate Temperatures

固定比较：

```text
tau = 1.00   # P0 baseline
tau = 0.20
tau = 0.10
tau = 0.05
tau = 0.02
```

不得看到结果后增加任意新的 tau 来追最优指标。

若所有候选 FAIL，则本轮 FAIL，并创建新的独立 decision 讨论 localized / top-k pooling。

---

# 7. Soft Center

对每个 frame：

```text
w_i = softmax(s_i / tau)

x_soft = sum(w_i * x_i)
y_soft = sum(w_i * y_i)
```

其中 patch coordinate 必须使用与 M0/P0 相同的坐标定义。

---

# 8. Global Object Token

Global token 同样使用：

```text
T_global = sum(w_i * z_i)
```

不得：

- hard argmax；
- binary mask；
- detector box crop；
- manual crop at inference。

---

# 9. Local Object Tokens

仍保持：

```text
2 × 2
```

四个 local regions。

每个 region 内：

```text
w_i_local = softmax(s_i / tau)
```

仅在该 region 的 patches 上归一化。

保持：

```text
semantic + x + y + confidence
```

不得改变 token 数和布局。

---

# 10. Confidence Definition

必须明确并固定 confidence 计算方式。

推荐：

```text
confidence_global = similarity-weighted concentration / peak statistic
```

或沿用 M0/P0 已有定义。

本轮不得同时引入新的 learnable confidence network。

---

# 11. Train-Only Calibration

严格使用 P0 已冻结的：

```text
8 reference frames
12 calibration-check frames
```

Dev 和 Final Test：

```text
0 reads
```

不得因为本轮是“修 bug”就查看 Dev/Test。

---

# 12. Primary Diagnostics

对每个 tau 报告：

## 12.1 Soft Center Error

```text
median
P90
mean
```

单位：

- pixels；
- normalized image diagonal；
- relative to coarse-box diagonal。

## 12.2 Tracking Behavior

分别比较：

```text
predicted x_soft vs target cup center x
predicted y_soft vs target cup center y
```

至少报告：

- Pearson correlation；
- Spearman correlation；
- predicted position span；
- target position span。

目的：

> 防止只把误差降了一点，但 soft center 仍基本不移动。

## 12.3 Similarity Concentration

报告：

```text
inside weighted mass
outside weighted mass
```

以及：

```text
effective patch count / entropy
```

用于判断 tau 是否过于 diffuse 或退化成单 patch argmax。

## 12.4 Existing Safety Checks

必须继续保持：

```text
peak_in_box_rate
cup_beats_corner_rate
```

---

# 13. Selection Rule

正式 tau 必须使用以下顺序选择。

首先排除任何：

```text
peak_in_box_rate < 1.0
cup_beats_corner_rate < 1.0
non-finite token
```

的候选。

剩余候选中：

> 选择满足 PASS 条件的最大 tau。

理由：

- 尽量保留 soft representation；
- 不无必要地退化成近似 argmax；
- 减少对反光单 patch 的敏感性。

---

# 14. PASS Criteria

本轮 PASS 必须同时满足：

### P0.1-G1 — Spatial Error

在 12 个 calibration-check frames 上：

```text
median soft-center error
<= 0.30 × median coarse-box diagonal
```

且：

```text
P90 soft-center error
<= 0.45 × median coarse-box diagonal
```

### P0.1-G2 — Position Tracking

预测 soft center 必须随杯位发生明显变化。

至少要求：

```text
x tracking direction = correct
y tracking direction = correct
```

并报告 Pearson / Spearman。

不得出现：

```text
predicted center span << target center span
```

的固定中心退化。

### P0.1-G3 — Object Peak

```text
peak_in_box_rate = 1.0
```

### P0.1-G4 — Shortcut

```text
cup_beats_corner_rate = 1.0
```

且 overlay 不显示桌面/角标主导。

### P0.1-G5 — Token Stability

5 tokens：

```text
finite
shape correct
x/y dynamic
confidence finite
local tokens not all identical
```

### P0.1-G6 — Scope Integrity

```text
Dev used = false
Final Test used = false
```

---

# 15. Over-Concentration Diagnostic

即使 center error PASS，也必须检查 tau 是否过小导致：

```text
one-patch collapse
```

报告：

- weight entropy；
- effective number of patches；
- top-1 weight；
- top-3 weight mass。

如果某 tau 基本退化为 hard argmax，而更大的 tau 已满足 PASS：

> 必须选择更大的 PASS tau。

---

# 16. Required Visualizations

每个候选 tau 至少生成代表性 overlay：

```text
RGB
+
similarity heatmap
+
coarse cup box
+
target box center
+
predicted soft center
+
peak patch
```

重点展示：

- 左；
- 中；
- 右；
- 前/后；
- 最大误差 frame；
- 最小误差 frame。

另外生成一张：

```text
target center trajectory
vs
predicted soft-center trajectory
```

用于直接判断是否跟随杯位。

---

# 17. Artifact Policy

P0 的：

```text
cup_prototype_bank_v1.pt
```

保持不变。

如果 P0.1 PASS，新产生的正式 artifact 应是：

```text
object_pooling_config_v1.json
```

至少记录：

```text
temperature
similarity rule
pooling rule
token layout
selection metrics
prototype sha256
git commit
```

Prototype Bank 本体不要重新命名成新 bank，避免错误暗示 prototype 被修改。

---

# 18. Result Status

### PASS

说明：

> Prototype + temperature-scaled soft pooling 已具备可用于正式 OC-ACT training 的空间表示能力。

### PARTIAL

例如：

- x 方向显著改善但 y 仍固定；
- center error 降低但未达到 gate；
- tau 太小才勉强工作，表现出严重 one-patch collapse。

PARTIAL 不得自动训练。

### FAIL

所有固定 tau 都不能解决 soft-center tracking。

下一步应创建新 decision，考虑：

```text
localized soft pooling
or
top-k neighborhood pooling
```

不得在本轮临时加入。

---

# 19. Required Deliverables

```text
experiments/position/p0_1_spatial_pooling_fix/index1/
├── decision.md
└── result/
    ├── summary.md
    ├── validation.json
    ├── metrics.json
    ├── object_pooling_config_v1.json      # only if PASS
    ├── figures/
    │   ├── tau_comparison.*
    │   ├── soft_center_tracking.*
    │   ├── representative_overlays/
    │   └── concentration_analysis.*
    └── proposal.md                        # optional
```

---

# 20. Codex MAY

- 修改 object pooling temperature；
- 编写 tau comparison；
- 重新生成 train-only diagnostics；
- 更新 object tokenizer 中对应 temperature config；
- 生成图表；
- 创建 proposal.md。

---

# 21. Codex MUST

- 保持 prototype bank 不变；
- 保持 reference/check set 不变；
- 严格 Train-only；
- 使用固定 tau candidates；
- 报告 tracking correlation/span；
- 检查 one-patch collapse；
- 完成后停止。

---

# 22. Codex MUST NOT

- 修改本 decision.md；
- 重新选择 prototype；
- 修改 prototype 数量；
- 使用 Dev / Final Test；
- 启动 policy training；
- fine-tune DINO；
- 引入 detector / SAM；
- 改 OC-ACT 主架构；
- 临时加入 top-k/local-neighborhood 方案；
- 启动 real robot；
- 自动进入正式训练。

---

# 23. Completion Rule

只有 P0.1：

```text
PASS
```

后才能标记：

```text
OBJECT_REPRESENTATION_READY = TRUE
```

随后停止。

下一阶段由用户 + ChatGPT 创建正式训练合同：

```text
T0 — ACT Baseline Training
T1 — OC-ACT Training
```
