# Experiment Decision Contract

> Revision: r0  
> 本文件由用户与 ChatGPT 冻结，Codex 只读。  
> 本 decision 用于一次性预注册夜间 8–9 小时 Discovery 批次。  
> Codex 可顺序或并行执行其中全部候选，但不得依据前一个候选结果修改后一个候选。

# N0 — Overnight Spatial Conditioning Discovery Batch

## 1. Identity

- **Experiment ID**: `N0_OVERNIGHT_SPATIAL_CONDITIONING_BATCH`
- **Track**: `POSITION`
- **Stage**: `DISCOVERY_ARCHITECTURE_BATCH`
- **Mode**: `DISCOVERY`
- **Primary Goal**: 在用户无法使用服务器的 8–9 小时内，高效预注册并训练多个彼此独立的 spatial-conditioning 候选。
- **Final Test**: 禁止
- **Real Robot**: 禁止
- **Per Candidate Seed Count**: `1`
- **Candidates**:
  - `R2_ACTION_LEVEL_SPATIAL`
  - `R1_RELATIONAL_STATE_SPATIAL`
  - `R3_SPATIAL_FILM`

已执行或正在执行的：

```text
T3_JOINT_STATE_SPATIAL_CONDITIONING
```

保持独立，不在本 decision 中修改。

---

# 2. Scientific Motivation

现有证据：

```text
Object branch is used.
Longitudinal steering is repeatedly strong.
Horizontal steering repeatedly fails.
```

已失败路线：

```text
absolute x/y as weak object-token fields
→ insufficient

independent SpatialMLP residual + trainable beta
→ path scale starvation / optimization rejection

fixed residual scale λ=0.5
→ path active but horizontal steering still fails
```

因此本批次不再继续搜索：

```text
beta
lambda
coordinate scale
```

而是比较三种更本质的 spatial-conditioning 机制：

```text
R2: action-level conditioning
R1: relational state-target conditioning
R3: feature-wise modulation (FiLM)
```

---

# 3. Common Frozen Upstream

全部候选必须冻结：

```text
Dataset = formal4/kind_merged_nomaster
Train = frozen 36 episodes
Dev = frozen 12 episodes
Final Test = frozen 12 episodes, DO NOT READ

Normalization = D1 train-only frozen normalization

DINO = dinov2_vits14
DINO frozen = true

Prototype bank = cup_prototype_bank_v1.pt
Prototype count = 4

Pooling config = object_pooling_config_v1.json
tau = 0.20

Object tokens = 1 global + 4 local
Object adapter = frozen architecture
Cross-attention = existing OC-ACT design
Original ACT Top branch = unchanged
Original ACT Wrist branch = unchanged
Robot state = 8D NOMASTER
Action = 6D
```

除各候选明确授权的单变量外，其余全部保持一致。

---

# 4. Common Spatial Source

全部 spatial condition 只允许来自：

```text
[x_global, y_global, confidence_global]
```

由 P0.1 frozen object pooling 产生。

禁止：

- GT box center
- detector box
- hand-entered position
- Dev/Test annotation
- new estimator
- post-hoc coordinate relabeling

---

# 5. Common Training Budget

每个候选：

```text
100,000 optimizer steps
1 seed
checkpoint every 10,000
Dev evaluation every 5,000
```

如训练框架无法严格实现 cadence：

> 在训练前写入真实 cadence，并保证所有候选一致。

同一批次候选必须尽可能使用：

```text
same seed
same optimizer
same LR
same LR schedule
same batch size
same action chunk
same augmentation
same preprocessing
same ACT initialization policy
same training budget
```

---

# 6. Baseline Policy

夜间批次为了节省 GPU 时间：

> 不要求为每个候选重新训练一个 baseline。

统一参照：

```text
当前已冻结、健康的 OC-ACT baseline checkpoint / result
```

如果当前工程中已有同一数据/split/normalization/seed 下的正式 OC-ACT baseline：

```text
reuse it
```

禁止为了提升某候选表现而选择不同 baseline。

若现有 baseline 与本批次 seed/配置无法公平比较：

> 仍可完成候选 Discovery，但必须在 summary 中标记：
> `BASELINE_COMPARISON_LIMITED = TRUE`

不得因此停止整批夜间训练。

---

# 7. Candidate Execution Priority

若单 GPU：

```text
1. R2_ACTION_LEVEL_SPATIAL
2. R1_RELATIONAL_STATE_SPATIAL
3. R3_SPATIAL_FILM
```

若多 GPU：

```text
可并行执行
```

禁止：

> R2 跑完后根据 R2 结果修改 R1/R3。

三个候选在启动第一个训练之前即视为全部冻结。

---

# 8. Candidate R2 — Action-Level Spatial Conditioning

## 8.1 Hypothesis

当前主要失败是：

> spatial information 无法可靠传递到 action steering。

因此直接在 ACT decoder / action-generation pathway 注入目标位置条件。

## 8.2 Frozen Encoder

Encoder 与现有 OC-ACT 完全一致。

不改变：

```text
state token
visual tokens
object cross-attention
object gate
encoder
```

## 8.3 Authorized Intervention

新增：

```text
[x,y,confidence]
      ↓
SpatialActionMLP
      ↓
spatial_action_embedding
```

然后融合到 ACT decoder query / action-generation representation。

推荐固定实现：

```text
decoder_query' = decoder_query + spatial_action_embedding
```

SpatialActionMLP：

```text
3
→ Linear(3, d_spatial)
→ LayerNorm
→ GELU
→ Linear(d_spatial, d_model)
```

其中：

```text
d_spatial = min(128, d_model)
```

## 8.4 No Gate

禁止再加：

```text
beta
lambda search
```

使用固定 residual：

```text
decoder_query' = decoder_query + spatial_action_embedding
```

如需要数值匹配，仅允许模块内部标准 LayerNorm，不允许额外人工 scale。

## 8.5 Forbidden

R2 不得：

- 修改 encoder
- 修改 object tokens
- 修改 state projection
- 修改 cross-attention
- 加 auxiliary loss
- 改 action head
- 改 DINO

---

# 9. Candidate R1 — Relational State-Spatial Conditioning

## 9.1 Hypothesis

单纯 absolute `[state,x,y]` 可能不足。

更合理的是显式学习：

> 当前机器人状态与目标位置之间的关系。

## 9.2 State Encoder

固定：

```text
s = StateEncoder(8D state)
```

输出：

```text
d_rel
```

## 9.3 Target Encoder

固定：

```text
t = TargetEncoder([x,y,confidence])
```

输出同维：

```text
d_rel
```

推荐：

```text
d_rel = min(128, d_model)
```

## 9.4 Relational Feature

固定构造：

```text
r = concat(
    s,
    t,
    s - t,
    s * t
)
```

然后：

```text
r
→ RelationalProjection
→ d_model
→ relational state-conditioning token
```

该 token 直接替代 original state token。

## 9.5 Important

这是本候选唯一结构变化。

禁止同时：

- 增加新的 spatial token
- 增加 FiLM
- 增加 decoder conditioning
- 改 object branch

---

# 10. Candidate R3 — Spatial FiLM Conditioning

## 10.1 Hypothesis

独立 additive residual 可能容易被模型忽略。

改用 feature-wise modulation：

```text
h' = gamma(x,y,c) ⊙ h + beta(x,y,c)
```

强制 spatial condition 调制已有 state representation。

## 10.2 Base Representation

使用原：

```text
state_token = StateProjection(8D state)
```

## 10.3 FiLM Generator

固定：

```text
[x,y,confidence]
→ FiLM MLP
→ gamma, beta
```

其中：

```text
gamma dimension = d_model
beta dimension = d_model
```

初始化要求：

```text
gamma ≈ 1
beta ≈ 0
```

使初始模型接近原 baseline。

## 10.4 Fusion

固定：

```text
state_token' = gamma * state_token + beta
```

然后进入原 ACT encoder。

## 10.5 Forbidden

R3 不得：

- 修改 decoder
- 修改 object branch
- 添加额外 token
- 搜索多个 FiLM layer
- 在多个网络层同时 FiLM

只允许：

```text
state token single-point FiLM
```

---

# 11. Common Training Health

所有候选必须记录：

- train loss
- Dev loss
- gradient norm
- candidate-module grad norm
- object adapter grad
- cross-attention grad
- DINO grad
- NaN/Inf
- GPU peak memory
- training throughput

要求：

```text
candidate module grad != 0
DINO grad = none/zero
```

---

# 12. Common Checkpoint Selection

每个候选：

> 使用 Dev primary action metric 最优 checkpoint。

tie：

> 选择更早 checkpoint。

禁止看 mechanism test 后重新挑 checkpoint。

---

# 13. Common Offline Metrics

全部候选至少报告：

```text
Dev inference L1
h0 L1
arm-only L1
gripper-only L1
per-action-dimension L1
position-stratified error
```

---

# 14. Frozen Spatial Mechanism Mapping

沿用此前预注册 mapping：

```text
horizontal primary action dim = dim0
longitudinal primary action dim = dim2
```

不得根据候选结果重新选择。

---

# 15. Common Counterfactual Tests

每个候选的 best checkpoint 必须执行。

## C0 — Normal

正常空间条件。

## C1 — Spatial Zero

```text
x=y=confidence=0
```

## C2 — Spatial Shuffle

跨 sample 打乱：

```text
x,y,confidence
```

## C3 — X Transplant

仅替换 donor x。

## C4 — Y Transplant

仅替换 donor y。

## C5 — Full Spatial Transplant

替换：

```text
x,y,confidence
```

---

# 16. Object-Token Transplant Preservation

继续复用已有：

```text
whole object transplant
local-only transplant
global-only transplant
```

目的：

> 检查新 spatial mechanism 是否破坏已有 object-conditioned longitudinal behavior。

---

# 17. Horizontal Discovery Criterion

重点指标：

```text
horizontal directional consistency
```

候选认为“成功”需要：

```text
>= 0.70
```

至少在以下一种稳定成立：

```text
X transplant
Full spatial transplant
Whole object transplant
```

同时：

- sign 与 frozen expert dim0 mapping 一致
- large-position > same-position control
- arm response 为主
- 非 gripper-only

---

# 18. Longitudinal Preservation Criterion

要求：

```text
longitudinal directional consistency >= 0.70
```

至少在：

```text
Y transplant
Full spatial transplant
Whole object transplant
```

之一成立。

---

# 19. Spatial Usage Criterion

必须证明候选模块不是形式存在。

至少满足：

```text
Normal vs Spatial Zero
```

或：

```text
Normal vs Spatial Shuffle
```

产生明显高于 numerical noise 的 arm-action change。

不能再次出现：

```text
~1e-7
```

级别。

---

# 20. Candidate PASS / PARTIAL / FAIL

## PASS

同时满足：

1. Training health PASS
2. candidate module receives gradients
3. horizontal steering >= 0.70
4. longitudinal steering >= 0.70
5. spatial counterfactual affects arm action
6. overall Dev action quality not clearly worse than frozen OC-ACT reference
7. Final Test untouched

## PARTIAL

例如：

- horizontal 0.60–0.70
- horizontal PASS but longitudinal weakens
- mechanism good but Dev L1 slightly worse

## FAIL

例如：

- horizontal near chance
- candidate module unused
- only gripper changes
- severe Dev degradation

---

# 21. Batch Ranking Rule

全部候选完成后，不自动选最终 winner。

Codex 只生成 ranking summary。

排序优先级固定：

```text
1. Horizontal mechanism
2. Longitudinal preservation
3. Overall Dev action quality
4. Mechanism cleanliness
5. Resource cost
```

禁止只按最低 Dev L1 排名。

---

# 22. No Architecture Search Inside Candidate

禁止：

```text
multiple hidden sizes
multiple fusion points
multiple lambda
multiple FiLM layers
multiple decoder injection variants
```

每个候选只有一个冻结版本。

夜间批次目标：

> broad mechanism screening

不是：

> hyperparameter optimization

---

# 23. Stop / Continue Rule

如果单 GPU 且服务器时间不足：

优先完成：

```text
R2
then R1
then R3
```

如果某个模型训练失败属于纯工程问题：

- 可自动修复明显工程 bug
- 允许重启该候选
- 不允许改变科研定义

如果是科研 FAIL：

> 继续执行后续候选，不得提前停止批次。

---

# 24. Final Test Protection

整个 batch 严格禁止：

```text
Final Test inference
Final Test loss
Final Test visualization
Final Test checkpoint selection
Final Test prototype use
```

---

# 25. No Real Robot

整个 batch：

```text
real robot = forbidden
```

---

# 26. Required Directory Layout

```text
experiments/position/n0_overnight_batch/index1/
├── decision.md
└── result/
    ├── batch_summary.md
    ├── batch_validation.json
    ├── ranking.json
    │
    ├── R2_action_level/
    │   ├── summary.md
    │   ├── metrics.json
    │   ├── training_manifest.json
    │   ├── checkpoints.json
    │   ├── mechanism.json
    │   ├── transplant_results.json
    │   └── figures/
    │
    ├── R1_relational/
    │   ├── summary.md
    │   ├── metrics.json
    │   ├── training_manifest.json
    │   ├── checkpoints.json
    │   ├── mechanism.json
    │   ├── transplant_results.json
    │   └── figures/
    │
    ├── R3_film/
    │   ├── summary.md
    │   ├── metrics.json
    │   ├── training_manifest.json
    │   ├── checkpoints.json
    │   ├── mechanism.json
    │   ├── transplant_results.json
    │   └── figures/
    │
    └── proposal.md
```

Checkpoints 可继续保存到：

```text
results/experiments/N0_OVERNIGHT_SPATIAL_CONDITIONING_BATCH/
```

---

# 27. Required Batch Summary Table

Codex 完成后必须生成类似：

```text
Model | Dev L1 | Horizontal | Longitudinal | Spatial Usage | Params | VRAM | Status
```

至少包含：

```text
T3        # if result already available
R2
R1
R3
Frozen OC-ACT reference
```

若 T3 尚无结果：

> 不等待 T3，先完成本 batch。

---

# 28. Codex MAY

- 顺序或并行训练 R2/R1/R3
- 使用 Train/Dev
- 自动继续下一个候选
- 运行冻结 counterfactual
- 生成图表
- 生成 ranking summary
- 修复纯工程 bug
- 创建 proposal.md

---

# 29. Codex MUST

- 在训练第一个候选前冻结三个候选配置
- 使用相同数据边界
- 使用相同 seed
- 使用相同训练预算
- 不根据前一个结果修改后一个模型
- 不读 Final Test
- 完成能执行的全部候选
- 最后统一生成 batch summary
- 完成后停止

---

# 30. Codex MUST NOT

- 修改本 decision.md
- 根据中途结果新增第四/第五候选
- 调 tau
- 重建 prototype
- fine-tune DINO
- 做 lambda search
- 做 hidden-size search
- 做 fusion-point search
- 加 auxiliary loss
- 使用 Final Test
- 启动 real robot
- 自动开始 multi-seed Confirmation

---

# 31. Completion Rule

夜间批次完成后：

```text
OVERNIGHT_DISCOVERY_BATCH_COMPLETE = TRUE
```

Codex 停止。

由用户 + ChatGPT 统一验收并决定：

```text
best candidate PASS
→ freeze candidate
→ C0 Position Offline Confirmation (>=3 seeds + Final Test)

only PARTIAL candidates
→ select one targeted follow-up

all FAIL
→ stop absolute/weak spatial-conditioning search
→ redesign representation / objective
```
