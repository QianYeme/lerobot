# Experiment Decision Contract

> Revision: r0  
> 本文件由用户与 ChatGPT 冻结，Codex 只读。

# T3 — Joint State-Spatial Conditioning Discovery

## 1. Identity

- **Experiment ID**: `T3_JOINT_STATE_SPATIAL_CONDITIONING`
- **Track**: `POSITION`
- **Stage**: `DISCOVERY_ARCHITECTURE_INTERVENTION`
- **Mode**: `DISCOVERY`
- **Upstream**:
  - `T1_OCACT_EXPLICIT_SPATIAL_CONDITIONING = FAIL`
  - `T1_1_EXPLICIT_SPATIAL_PATH_HEALTH = PASS_DIAGNOSIS`
  - `T2_FIXED_SCALE_SPATIAL_CONDITIONING = FAIL`
- **Final Test**: 禁止
- **Real Robot**: 禁止
- **Seed Count**: `1`

本轮只允许一个科研干预：

> 不再把 `(x,y,confidence)` 作为独立 residual 加到 state token；改为将 8D robot state 与 3D target spatial condition 直接拼接，并通过一个联合投影生成唯一的 state-conditioning token。

除此之外全部冻结。

## 2. Frozen Evidence

T1 中 trainable beta 路径被压到近乎关闭：

```text
beta_final ≈ 0.00318
```

T1.1 诊断：

```text
primary = SCALE_STARVATION
secondary = OPTIMIZATION_REJECTION
```

T2 固定 `lambda=0.5` 后，空间路径数值上确实活跃：

```text
Normal vs Spatial Zero
Dev L1 change ≈ 0.002868
```

但机制仍失败：

```text
horizontal directional consistency = 0.10
longitudinal directional consistency = 0.90
```

且：

```text
B2 Dev L1 = 0.088282
P3 Dev L1 = 0.097128
```

因此当前结论是：

> 单纯“独立 spatial residual + 更大 scale”不能解决横向 steering。

## 3. Research Question

> **RQ:** 如果目标位置不是独立附加到 state token，而是与 robot state 从输入层开始联合编码，模型能否学习“当前机械臂状态相对于目标位置”的条件关系，从而恢复可靠的横向 steering，并保持纵向机制与整体动作质量？

## 4. Compared Models

### B3 — Current OC-ACT

```text
8D robot state
→ original StateProjection
→ state token
```

原 object branch 保持：

```text
Top RGB
→ Frozen DINOv2-S
→ 4-prototype bank
→ tau=0.20 pooling
→ 1 global + 4 local tokens
→ adapter
→ pre-encoder cross-attention
→ gated residual
→ ACT
```

### P4 — OC-ACT + Joint State-Spatial Token

唯一变化：

```text
[8D robot state,
 x_global,
 y_global,
 confidence_global]
        ↓
Joint State-Spatial Projection
        ↓
single state-conditioning token
```

不再存在：

```text
separate SpatialMLP residual
beta
lambda
```

## 5. Frozen Spatial Source

固定使用：

```text
x_global
y_global
confidence_global
```

来源：

```text
P0.1 frozen object pooling
prototype count = 4
tau = 0.20
```

禁止使用 GT box center、detector box、Dev/Test annotation、hand-entered coordinates 或新的 position estimator。

## 6. Input Representation

联合输入固定为：

```text
11D = 8D robot state + 3D target spatial condition
```

其中：

```text
x,y ∈ [-1,1]
```

confidence 使用 P0.1 已冻结定义。

## 7. Joint Projection

第一版固定采用轻量 MLP：

```text
11
↓
Linear(11, d_joint)
↓
LayerNorm
↓
GELU
↓
Linear(d_joint, d_model)
```

其中：

```text
d_joint = min(128, d_model)
```

输出：

```text
joint_state_token
```

直接替代原来的：

```text
StateProjection(8D state)
```

## 8. No Additional Spatial Gate

P4 禁止：

```text
beta
lambda
spatial residual
```

空间信息通过 joint projection 自然参与 state-condition 表示。

## 9. No Additional Spatial Token

本轮不新增独立 spatial token。

目的是避免重现旧 C0 独立 box/spatial token 被大量 visual tokens 淹没的问题。

## 10. Frozen Components

以下全部不变：

```text
Train / Dev split
Normalization
DINOv2-S
DINO frozen
Prototype bank
Prototype count = 4
tau = 0.20
5 object tokens
Object adapter
Cross-attention
Object gate alpha
ACT Top backbone
ACT Wrist backbone
ACT encoder
ACT decoder
Action head
6D action
```

唯一主要科研差异：

```text
StateProjection(8D)
vs
JointStateSpatialProjection(11D)
```

## 11. Initialization Policy

P4 的 Joint Projection 默认从头初始化。

禁止人为放大 x/y 权重、给 x/y 单独 scale、使用 Fourier features 或额外 positional encoding。

## 12. Training Fairness

B3/P4 必须一致：

- Train 36；
- Dev 12；
- train-only normalization；
- same seed；
- same optimizer；
- same LR；
- same LR schedule；
- same batch size；
- same image preprocessing；
- same action chunk；
- same ACT backbone init policy；
- same steps；
- same checkpoint cadence；
- same Dev eval cadence。

## 13. Training Budget

固定：

```text
100,000 optimizer steps
checkpoint every 10k
Dev evaluation every 5k
```

如框架实际 cadence 有限制，需在训练前写入 manifest，且 B3/P4 完全一致。

## 14. Seed

固定：

```text
1 seed
```

B3/P4 使用相同 seed。

## 15. Checkpoint Selection

预先冻结：

> 使用 Dev primary action metric 最优 checkpoint。

tie 时选择更早 checkpoint。

禁止根据 transplant / mechanism 结果重选 checkpoint。

## 16. Training Health

必须记录：

- train loss；
- Dev loss；
- global grad norm；
- Joint Projection grad norm；
- object adapter grad；
- cross-attention grad；
- alpha；
- DINO grad；
- NaN/Inf。

P4 要求：

```text
Joint Projection grad != 0
```

DINO：

```text
grad = none/zero
```

## 17. Primary Offline Performance

B3/P4 同样报告：

- Dev inference L1；
- h0 L1；
- arm-only L1；
- gripper-only L1；
- per-action-dim L1；
- position-stratified error。

## 18. Spatial Mechanism Evaluation

复用 T0.2/T2 已冻结 action mapping。

### Horizontal

```text
primary action dim = dim0
```

### Longitudinal

```text
primary action dim = dim2
```

不得看 P4 结果后更换。

## 19. Joint-Condition Counterfactuals

### J0 — Normal

正确 state + x/y/conf。

### J1 — Spatial Zero

保持 state：

```text
x=y=confidence=0
```

### J2 — XY Shuffle

state 保持当前 frame，仅打乱：

```text
x,y,confidence
```

### J3 — X Transplant

state 不变，只替换 donor：

```text
x
```

保留 base y/confidence。

### J4 — Y Transplant

state 不变，只替换 donor：

```text
y
```

保留 base x/confidence。

### J5 — Full Spatial Transplant

state 不变，替换 donor：

```text
x,y,confidence
```

## 20. Object Token Transplant

继续保留：

```text
whole object tokens
local-only
global-only
```

用于检查 joint spatial token 是否与原 object branch 形成互补，而不是破坏已有纵向机制。

## 21. Horizontal PASS Criteria

至少要求：

```text
horizontal directional consistency >= 0.70
```

在以下至少一种成立：

```text
J3 X-transplant
J5 full-spatial transplant
whole-object transplant
```

且：

- sign 与 expert dim0 mapping 一致；
- large-position response > same-position control；
- arm action 为主要变化；
- 不是只改变 gripper。

## 22. Longitudinal Preservation

要求：

```text
longitudinal directional consistency >= 0.70
```

在：

```text
J4 / J5 / whole object transplant
```

至少一个稳定成立。

## 23. Joint Spatial Usage

必须证明 joint token 真的利用 spatial 部分。

要求：

```text
Normal vs Spatial Zero
```

或：

```text
Normal vs XY Shuffle
```

产生明显高于数值噪声的 arm-action 差异。

同时：

```text
X transplant
```

应主要影响 horizontal-sensitive action；

```text
Y transplant
```

应主要影响 longitudinal-sensitive action。

## 24. Overall Discovery PASS

### PASS

同时满足：

1. Training health 正常；
2. Joint Projection 有梯度；
3. horizontal steering >= 0.70；
4. longitudinal steering >= 0.70；
5. spatial counterfactual 明确影响 arm action；
6. P4 Dev action quality 不明显差于 B3；
7. Final Test 未使用。

### PARTIAL

例如：

- horizontal 0.60–0.70；
- 横向 PASS，但 longitudinal 下降；
- mechanism 很好但 Dev L1 小幅退化。

### FAIL

例如：

- joint token 仍忽略 x/y；
- horizontal 仍接近 chance；
- action quality 明显下降；
- 只有 gripper 响应。

## 25. Interpretation

如果 PASS：

> 支持“机器人状态与目标位置需要联合条件编码，而不是独立 spatial residual”这一机制解释。

如果 FAIL：

> 不再继续简单 absolute `(x,y)` conditioning 路线；下一步转向 relative target-state representation。

## 26. Final Test Protection

禁止：

- Final Test inference；
- Final Test loss；
- Final Test visualization；
- Final Test model selection。

## 27. No Real Robot

本轮禁止真机。

## 28. Required Outputs

```text
experiments/position/t3_joint_state_spatial/index1/
├── decision.md
└── result/
    ├── summary.md
    ├── validation.json
    ├── metrics.json
    ├── training_manifest.json
    ├── checkpoints.json
    ├── mechanism.json
    ├── joint_counterfactual.json
    ├── transplant_results.json
    ├── resources.json
    ├── figures/
    │   ├── train_dev_loss.*
    │   ├── horizontal_directionality.*
    │   ├── longitudinal_directionality.*
    │   ├── joint_zero_shuffle.*
    │   ├── x_transplant.*
    │   ├── y_transplant.*
    │   └── per_action_dimension.*
    └── proposal.md
```

## 29. Codex MAY

- 实现 JointStateSpatialProjection；
- 训练 B3/P4；
- 使用 Train/Dev；
- 执行冻结 counterfactual；
- 执行 token transplant；
- 做统计与可视化；
- 做纯工程 bugfix；
- 创建 proposal.md。

## 30. Codex MUST

- 只改变 state/spatial fusion 方式；
- 不增加 spatial gate；
- 不改其它 OC-ACT 结构；
- B3/P4 公平训练；
- same seed；
- 不读 Final Test；
- 使用冻结 action mapping；
- 完成后停止。

## 31. Codex MUST NOT

- 修改本 decision.md；
- 搜索多个 joint architectures；
- 改 prototype；
- 调 tau；
- fine-tune DINO；
- 改 object token count；
- 改 cross-attention；
- 人为放大 x/y；
- 加 auxiliary loss；
- 加新 detector；
- 使用 Final Test；
- 启动 real robot；
- 自动进入 Confirmation。

## 32. Completion Rule

结果由用户 + ChatGPT 验收：

```text
PASS
→ 冻结 P4
→ C0 Position Offline Confirmation (>=3 seeds + Final Test)

PARTIAL
→ 仅在证据充分时再考虑一个单变量 Discovery

FAIL
→ 停止 absolute spatial-conditioning 路线
→ 转向 relative target-state representation
```

本轮结束后停止。
