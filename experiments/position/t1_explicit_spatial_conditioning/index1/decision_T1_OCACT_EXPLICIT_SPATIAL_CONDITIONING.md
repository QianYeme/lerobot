# Experiment Decision Contract

> Revision: r0
> 本文件由用户与 ChatGPT 冻结，Codex 只读。

# T1 — OC-ACT Explicit Spatial Conditioning Discovery

## 1. Identity

- **Experiment ID**: `T1_OCACT_EXPLICIT_SPATIAL_CONDITIONING`
- **Track**: `POSITION`
- **Stage**: `DISCOVERY_ARCHITECTURE_INTERVENTION`
- **Mode**: `DISCOVERY`
- **Upstream**:
  - `T0_POSITION_DISCOVERY_ACT_VS_OCACT = PARTIAL`
  - `T0_1_OCACT_SPATIAL_MECHANISM_DIAGNOSTIC = PARTIAL`
  - `T0_2_OCACT_TOKEN_TRANSPLANT_DIRECTIONALITY = PARTIAL`
- **Formal Confirmation**: 禁止
- **Final Test**: 禁止
- **Real Robot**: 禁止
- **Seed Count**: `1`

本轮只允许一个科研干预：

> 在当前 OC-ACT 上增加一条显式二维目标位置条件路径，将 cup global soft center `(x, y, confidence)` 经过轻量 Spatial MLP 后直接融合到 ACT state-conditioning token，使目标位置不能仅依赖 object token 内部弱坐标字段被注意力间接读取。

其它结构全部冻结。

---

# 2. Upstream Evidence

当前冻结 P1 OC-ACT 已知：

## 2.1 Performance

```text
ACT Dev L1 ≈ 0.241336
OC-ACT Dev L1 ≈ 0.230687
relative improvement ≈ 4.43%
```

## 2.2 Object Path

```text
zero object → 明显 action change
shuffle object → 明显 action change
```

说明：

> object branch 被 policy 使用。

## 2.3 Spatial Diagnostics

T0.1：

- local token layout 被使用；
- explicit numeric `(x,y)` 贡献很弱。

T0.2：

### Longitudinal

```text
whole directional consistency = 0.90
local-only = 0.85
global-only = 0.90
```

且 large-position response > same-position control。

因此：

> 纵向 object-conditioned steering 有机制证据。

### Horizontal

```text
whole = 0.20
local-only = 0.00
global-only = 0.20
```

因此：

> 当前模型没有可靠横向 steering 证据。

---

# 3. Research Question

> **RQ:** 在保持 DINO、prototype、object token、cross-attention 和 ACT 主干不变的前提下，将 global cup position 显式融合进 state-conditioning token，能否建立可靠的横向目标位置 steering，同时保留已有纵向 steering 与整体 action quality？

---

# 4. Compared Models

本轮只比较：

## B1 — Frozen Architecture OC-ACT

即 T0 中的 P1 架构：

```text
Object Tokens
→ Cross-Attention
→ Gated Residual
→ ACT
```

训练从头按同一预算训练一个新的 B1 discovery run，避免拿历史 checkpoint 与新 seed / 新训练环境直接比较。

---

## P2 — OC-ACT + Explicit Spatial Conditioning

唯一新增：

```text
global cup soft center:
[x, y, confidence]
        ↓
Spatial MLP
        ↓
d_model spatial embedding
        ↓
add / residual fuse into ACT state-conditioning token
        ↓
Original ACT Encoder
```

同时原来的：

```text
Object Tokens
→ Cross-Attention
→ Gated Residual
```

完全保留。

---

# 5. Frozen Components

以下全部保持不变：

```text
Dataset
Train / Dev split
Normalization
DINOv2-S
DINO frozen
Prototype bank = 4 prototypes
Pooling tau = 0.20
Object token count = 5
1 global + 4 local
Object token semantic representation
Cross-attention position
Cross-attention architecture
Object gate
ACT Top visual branch
ACT Wrist visual branch
ACT encoder / decoder
Action head
8D NOMASTER state
6D action
```

禁止同时改变其它科研结构。

---

# 6. Explicit Spatial Input

空间输入固定为：

```text
[x_global, y_global, confidence_global]
```

来源必须是现有 P0.1 frozen object pooling：

```text
temperature-scaled soft center
tau = 0.20
```

禁止：

- 新 detector box；
- GT box center；
- hand-entered position；
- Dev annotation；
- Final Test information。

---

# 7. Coordinate Representation

使用现有归一化坐标：

```text
x, y ∈ [-1, 1]
```

confidence 使用现有 frozen definition。

本轮禁止：

- 手工把 x 乘 10；
- 单独人为放大横轴；
- 看结果后调 coordinate scale；
- Fourier features；
- sinusoidal encoding；
- learnable lookup grid。

这些若以后需要，必须另开 decision。

---

# 8. Spatial MLP

固定为轻量结构：

```text
3
↓
Linear(3, d_spatial)
↓
LayerNorm
↓
GELU
↓
Linear(d_spatial, d_model)
```

推荐：

```text
d_spatial = min(128, d_model)
```

不得构造大网络。

---

# 9. Fusion Point

第一版固定：

> **融合到 ACT state-conditioning token**

形式：

```text
state_token = StateProjection(robot_state)

spatial_embedding = SpatialMLP([x,y,conf])

state_token' = state_token + beta * spatial_embedding
```

然后：

```text
state_token'
+
visual/object-conditioned tokens
→ Original ACT Encoder
```

---

# 10. Spatial Gate

增加单独的：

```text
beta
```

要求：

- trainable scalar；
- 初始化为小非零值。

固定建议：

```text
beta_init = 0.01
```

原因：

> 若初始化严格为 0，Spatial MLP 初始可能收不到有效梯度；本轮希望路径从开始即可学习，同时仍保持接近 baseline。

不得在本轮比较多个 beta_init。

---

# 11. Why State-Token Fusion

本设计不是创建一个新的独立 spatial token。

原因：

> 旧 C0 independent box token 曾出现被大量 visual tokens 淹没、几乎不影响 action 的问题。

本轮直接将目标位置融合到 state-conditioning token：

```text
robot configuration
+
target location
```

成为同一个条件表示。

科研假设：

> Reach action 本质上依赖“当前机械臂状态相对于目标位置”的联合条件，因此 state-token fusion 比独立弱坐标 token 更合理。

---

# 12. Training Fairness

B1 与 P2 必须使用完全一致：

- Train 36；
- Dev 12；
- train-only normalization；
- seed；
- optimizer；
- LR；
- LR schedule；
- batch size；
- action chunk；
- training steps；
- checkpoint cadence；
- eval cadence；
- image preprocessing；
- ACT initialization policy。

唯一主要差异：

```text
P2 has SpatialMLP + beta fusion
```

---

# 13. Training Budget

Discovery：

```text
100,000 optimizer steps
```

B1/P2 均：

```text
100k
```

固定：

```text
checkpoint every 10k
Dev evaluation every 5k
```

如当前训练框架实际不能严格实现该 cadence，应在启动前报告并保持 B1/P2 完全一致。

---

# 14. Seed

固定一个 Discovery seed。

必须在启动前写入：

```text
training_manifest.json
```

B1/P2 使用同 seed。

---

# 15. Checkpoint Selection

冻结：

> 使用 Dev primary action metric 最优 checkpoint。

若 tie，则优先更早 checkpoint。

不得看空间机制结果后重新挑 checkpoint。

---

# 16. Training Health

记录：

- train loss；
- Dev loss；
- gradient norm；
- NaN/Inf；
- alpha；
- beta；
- SpatialMLP grad norm；
- object adapter grad；
- cross-attention grad；
- DINO grad。

要求：

```text
DINO grad = none/zero
SpatialMLP grad != 0
beta grad != 0
```

---

# 17. Primary Offline Performance

B1/P2 同样报告：

- Dev inference L1；
- per-action-dim L1；
- h0 L1；
- arm-only；
- gripper-only；
- position-stratified diagnostics。

本轮目标不是必须极大降低总 L1。

---

# 18. Primary Spatial Mechanism Evaluation

必须复用 T0.2 的冻结诊断协议。

不得重新发明有利于 P2 的测试。

## 18.1 Horizontal Token Transplant

继续使用：

```text
primary action dim = dim0
```

测：

- whole-token transplant；
- local-only；
- global-only。

重点：

```text
horizontal directional consistency
```

---

## 18.2 Longitudinal Token Transplant

继续：

```text
primary action dim = dim2
```

确保新 intervention 不破坏已有纵向机制。

---

# 19. New Spatial-Path Ablations

P2 必须额外做：

## S0 — Normal

完整 spatial conditioning。

## S1 — Zero Spatial Embedding

```text
spatial_embedding = 0
```

但 object tokens 保持正常。

回答：

> 新 spatial path 是否实际被使用。

## S2 — Shuffle Spatial Coordinates

在 batch / matched frames 间只打乱：

```text
x,y,confidence
```

object tokens 仍保持当前 frame。

## S3 — X-only Counterfactual

替换：

```text
x
```

保留：

```text
y, confidence
```

检查横向 action 是否方向性变化。

## S4 — Y-only Counterfactual

用于确认纵向没有被异常耦合。

---

# 20. Horizontal PASS Criteria

P2 想进入 Confirmation，必须显著改善当前横向机制。

至少满足：

```text
horizontal directional consistency >= 0.70
```

在 whole-token / spatial-coordinate transplant 中至少有一项稳定成立。

并且：

- sign 与 expert dim0 mapping 一致；
- large-position response > same-position control；
- 不是 gripper-only；
- x-only counterfactual 有合理方向作用。

---

# 21. Longitudinal Preservation

P2 不得以修横向为代价破坏纵向。

要求：

```text
longitudinal directional consistency >= 0.70
```

或至少不显著低于 frozen P1 的可重复水平。

如果横向 PASS 但纵向彻底消失：

```text
PARTIAL
```

---

# 22. Overall Discovery PASS

### PASS

需要同时满足：

1. Training health PASS；
2. Explicit spatial path 有梯度且被使用；
3. Horizontal steering 从原 P1 的失败状态提升到可靠方向性；
4. Longitudinal steering 保持；
5. Dev overall action quality 不发生明显退化；
6. Final Test 未使用。

### PARTIAL

例如：

- 横向提高但仅 0.60–0.70；
- 横向 PASS 但纵向变弱；
- 机制改善明显但 overall action L1 小幅下降。

### FAIL

例如：

- beta / SpatialMLP 学不到；
- horizontal steering仍接近 chance；
- 改善只来自 gripper；
- action quality明显恶化；
- 需要人为放大 x 才工作。

---

# 23. Final Test Protection

本轮禁止任何 Final Test inference。

包括：

- loss；
- prediction；
- visualization；
- prototype；
- checkpoint selection。

---

# 24. No Real Robot

本轮禁止真机。

先完成 Discovery 机制闭环。

---

# 25. Required Outputs

```text
experiments/position/t1_explicit_spatial_conditioning/index1/
├── decision.md
└── result/
    ├── summary.md
    ├── validation.json
    ├── metrics.json
    ├── training_manifest.json
    ├── checkpoints.json
    ├── mechanism.json
    ├── transplant_results.json
    ├── resources.json
    ├── figures/
    │   ├── train_dev_loss.*
    │   ├── alpha_beta.*
    │   ├── horizontal_directionality.*
    │   ├── longitudinal_directionality.*
    │   ├── spatial_zero_shuffle.*
    │   ├── x_only_counterfactual.*
    │   └── per_action_dimension.*
    └── proposal.md
```

---

# 26. Codex MAY

- 实现 SpatialMLP；
- 实现 beta fusion；
- 训练 B1/P2；
- 使用 Train/Dev；
- 运行冻结的 transplant diagnostics；
- 做 S0-S4 spatial ablations；
- 生成图表；
- 做纯工程 bugfix；
- 创建 proposal.md。

---

# 27. Codex MUST

- 保持其它结构冻结；
- 只增加一个 explicit spatial path；
- B1/P2 公平训练；
- 使用相同 seed；
- 不读 Final Test；
- 复用 T0.2 action mapping；
- 报告横向与纵向独立结果；
- 完成后停止。

---

# 28. Codex MUST NOT

- 修改本 decision.md；
- 改 prototype；
- 调 tau；
- fine-tune DINO；
- 改 cross-attention；
- 改 object token count；
- 手工放大 x；
- 添加 positional loss；
- 添加 contrastive loss；
- 同时修改多个结构；
- 使用 Final Test；
- 启动 real robot；
- 自动进入 Confirmation。

---

# 29. Completion Rule

结果由用户 + ChatGPT 验收：

```text
PASS
→ 冻结 P2
→ C0 Position Offline Confirmation (>=3 seeds + Final Test)

PARTIAL
→ 根据证据决定是否还值得一个单变量 Discovery

FAIL
→ reject explicit state-token spatial fusion
```

本轮结束后停止。
