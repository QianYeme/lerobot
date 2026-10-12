# Experiment Decision Contract

> Revision: r0  
> 本文件由用户与 ChatGPT 冻结，Codex 只读。

# T2 — Fixed-Scale Explicit Spatial Conditioning Discovery

## 1. Identity

- **Experiment ID**: `T2_FIXED_SCALE_SPATIAL_CONDITIONING`
- **Track**: `POSITION`
- **Stage**: `DISCOVERY_ARCHITECTURE_INTERVENTION`
- **Mode**: `DISCOVERY`
- **Upstream**:
  - `T1_OCACT_EXPLICIT_SPATIAL_CONDITIONING = FAIL`
  - `T1_1_EXPLICIT_SPATIAL_PATH_HEALTH = PASS_DIAGNOSIS`
- **Final Test**: 禁止
- **Real Robot**: 禁止
- **Seed Count**: `1`

本轮只允许一个科研干预：

> 将 T1 中 trainable scalar `beta` 改为固定、不可训练的 residual scale，避免显式空间路径在优化过程中被压缩到近乎关闭。

除此以外全部保持不变。

---

## 2. Frozen Diagnosis

T1.1 已确认：

```text
primary diagnosis = SCALE_STARVATION
secondary = OPTIMIZATION_REJECTION
```

关键证据：

```text
learned beta ≈ 0.00318068

state token norm ≈ 24.4877
raw spatial embedding norm ≈ 8.5522
effective residual norm ≈ 0.0272

effective spatial / state
≈ 0.00111
≈ 0.11%
```

在 learned beta 下：

```text
mean action delta ≈ 1.27e-07
```

而临时 override：

```text
beta = 0.5
→ mean action delta ≈ 9.57e-04

beta = 1.0
→ mean action delta ≈ 4.15e-03
```

且在 `beta=1.0` 下：

```text
x flip
y flip
xy shuffle
zero spatial
```

均产生可测动作变化。

因此：

> 当前 explicit spatial representation 与 fusion path 并非完全无效；主要问题是 trainable beta 被优化到数值过小。

---

## 3. Research Question

> **RQ:** 在保持 spatial representation、SpatialMLP、fusion point 和 OC-ACT 主架构不变的情况下，仅将可学习 gate 改为固定残差尺度，能否让显式目标位置路径持续参与学习，并恢复可靠的横向 spatial steering？

---

## 4. Compared Models

本轮只比较：

### B2 — OC-ACT Baseline

使用当前冻结 OC-ACT，不含 explicit spatial path：

```text
DINO
→ prototype
→ object tokens
→ cross-attention
→ gated residual
→ ACT
```

### P3 — OC-ACT + Fixed-Scale Spatial Conditioning

保留 T1：

```text
[x,y,confidence]
→ SpatialMLP
→ spatial embedding
→ state token residual fusion
```

唯一变化：

```text
不再使用 trainable beta
```

改为：

```text
state_token' = state_token + λ * spatial_embedding
```

其中：

```text
λ = fixed constant
requires_grad = False
```

---

## 5. Fixed Scale

本轮固定：

```text
lambda = 0.5
```

理由：

T1.1 诊断中：

```text
beta=0.5
```

已经产生稳定、可测的 spatial sensitivity，同时比 `1.0` 更保守，降低过强残差破坏 ACT state representation 的风险。

本轮禁止比较多个 lambda。

禁止：

```text
0.1 / 0.2 / 0.5 / 1.0 grid search
```

如 `lambda=0.5` FAIL，下一轮另开 decision。

---

## 6. Frozen Spatial Representation

保持：

```text
[x_global, y_global, confidence_global]
```

来源：

```text
P0.1 frozen soft center
tau = 0.20
```

禁止：

- 改 coordinate estimator；
- 改 normalization；
- 改 prototype；
- 改 tau；
- 改 confidence；
- GT coordinate；
- Dev/Test annotation。

---

## 7. Frozen SpatialMLP

与 T1 完全相同：

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

不得：

- 加深；
- 加宽；
- 加 Fourier feature；
- 加 positional encoding；
- 加 auxiliary head。

---

## 8. Frozen Fusion Point

继续：

```text
state_token = StateProjection(robot_state)

spatial_embedding = SpatialMLP([x,y,confidence])

state_token' = state_token + 0.5 * spatial_embedding
```

再进入：

```text
Original ACT Encoder
```

禁止：

- 改为 concat；
- 改 fusion location；
- 改到 visual token；
- 新建独立 spatial token。

---

## 9. Training Fairness

B2 / P3 必须一致：

- Train 36；
- Dev 12；
- train-only normalization；
- same seed；
- same optimizer；
- same LR；
- same schedule；
- same batch size；
- same image preprocessing；
- same action chunk；
- same ACT initialization policy；
- same training steps；
- same checkpoint/eval cadence。

唯一主要差异：

```text
P3 has Fixed-Scale SpatialMLP residual
```

---

## 10. Training Budget

固定：

```text
100,000 optimizer steps
```

两模型都：

```text
checkpoint every 10k
Dev eval every 5k
```

如框架限制导致实际 cadence 不同：

> 必须在启动前记录，且 B2/P3 完全一致。

---

## 11. Seed

Discovery：

```text
1 seed
```

B2/P3 使用相同 seed。

---

## 12. Checkpoint Selection

冻结：

> Dev primary action metric 最优 checkpoint。

若并列：

> 选择更早 checkpoint。

不得看 steering 结果后重选 checkpoint。

---

## 13. Training Health

记录：

- train loss；
- Dev loss；
- grad norm；
- NaN/Inf；
- SpatialMLP grad norm；
- state projection grad；
- object adapter grad；
- cross-attention grad；
- DINO grad；
- effective spatial residual norm。

本轮没有 trainable beta。

必须确认：

```text
lambda = 0.5 fixed
SpatialMLP receives nonzero gradient
DINO remains frozen
```

---

## 14. Scale Monitoring

每个 eval checkpoint 至少报告：

```text
||state_token||
||spatial_embedding||
||0.5 * spatial_embedding||
ratio = effective spatial / state
```

目标不是追求某个固定比例，而是确保不再退化到：

```text
~0.1%
```

的 near-off 状态。

---

## 15. Offline Performance

B2/P3 同样报告：

- Dev inference L1；
- h0 L1；
- arm-only L1；
- gripper-only L1；
- per-action-dim L1；
- position-stratified error。

---

## 16. Primary Spatial Mechanism Tests

必须复用 T0.2 / T1 的冻结 protocol。

### Horizontal

primary:

```text
action dim0
```

测：

- whole object/token transplant；
- local-only；
- global-only；
- explicit spatial x counterfactual。

### Longitudinal

primary:

```text
action dim2
```

测：

- same set；
- y counterfactual。

---

## 17. Explicit Spatial Path Ablations

对 P3 固定 checkpoint：

### S0 — Normal

```text
lambda = 0.5
```

### S1 — Spatial Zero

推理时：

```text
spatial_embedding = 0
```

### S2 — XY Shuffle

只打乱：

```text
x,y,confidence
```

object tokens 保持原样。

### S3 — X Flip

```text
x -> -x
```

### S4 — Y Flip

```text
y -> -y
```

这些只用于机制诊断。

---

## 18. PASS Criteria

### Training Health

必须：

```text
no NaN/Inf
SpatialMLP grad != 0
effective residual remains numerically active
```

### Horizontal Mechanism

核心要求：

```text
horizontal directional consistency >= 0.70
```

至少在：

- token transplant；
- x-specific spatial counterfactual；

之一稳定成立。

同时：

- sign 与 frozen expert dim0 mapping 一致；
- large-position > same-position control；
- arm response 为主，不是 gripper-only。

### Longitudinal Preservation

要求：

```text
longitudinal directional consistency >= 0.70
```

或至少不显著低于原 OC-ACT 可重复水平。

### Spatial Path Usage

必须：

```text
Normal vs Spatial Zero
```

或：

```text
Normal vs XY Shuffle
```

产生明显高于数值噪声的 action change。

不能再次出现：

```text
~1e-7
```

级别近似无效。

### Overall Quality

P3 Dev action quality 不得明显劣于 B2。

---

## 19. PASS / PARTIAL / FAIL

### PASS

同时满足：

1. fixed spatial path 数值上持续活跃；
2. horizontal steering 达到可靠方向性；
3. longitudinal steering 保持；
4. Dev action quality 不明显退化；
5. Final Test 未使用。

### PARTIAL

例如：

- horizontal 提升到 0.60–0.70；
- horizontal PASS 但 Dev L1 小幅变差；
- spatial path 有效但纵向下降。

### FAIL

例如：

- SpatialMLP 仍被网络等效忽略；
- horizontal 仍接近 chance；
- fixed residual 明显破坏 action quality；
- 只有 gripper 变化；
- 必须继续提高 lambda 才有作用。

---

## 20. Interpretation

如果 PASS：

> 支持“显式目标位置路径此前失败主要源于可学习 gate 的 scale starvation，而不是 representation 本身无效”。

随后可冻结 P3 进入多 seed Confirmation。

如果 FAIL：

> 不再继续单纯通过扩大 residual scale 修复；下一步应考虑 fusion suppression / relative target-state representation 等新的单变量结构。

---

## 21. Final Test Protection

本轮禁止：

- Final Test inference；
- Final Test visualization；
- Final Test loss；
- Final Test checkpoint selection。

---

## 22. No Real Robot

本轮禁止真机。

---

## 23. Required Outputs

```text
experiments/position/t2_fixed_scale_spatial/index1/
├── decision.md
└── result/
    ├── summary.md
    ├── validation.json
    ├── metrics.json
    ├── training_manifest.json
    ├── checkpoints.json
    ├── mechanism.json
    ├── spatial_scale.json
    ├── transplant_results.json
    ├── figures/
    │   ├── train_dev_loss.*
    │   ├── spatial_residual_scale.*
    │   ├── horizontal_directionality.*
    │   ├── longitudinal_directionality.*
    │   ├── spatial_zero_shuffle.*
    │   └── per_action_dimension.*
    └── proposal.md
```

---

## 24. Codex MAY

- 将 trainable beta 改为固定 lambda=0.5；
- 训练 B2/P3；
- 使用 Train/Dev；
- 运行冻结 diagnostics；
- 生成图表；
- 做纯工程 bugfix；
- 创建 proposal.md。

---

## 25. Codex MUST

- 只修改 gate strategy；
- 保持 SpatialMLP / fusion / representation 不变；
- B2/P3 公平训练；
- 使用相同 seed；
- 记录 residual scale；
- 不读 Final Test；
- 完成后停止。

---

## 26. Codex MUST NOT

- 修改本 decision.md；
- 搜索多个 lambda；
- 修改 prototype；
- 调 tau；
- fine-tune DINO；
- 改 SpatialMLP；
- 改 fusion point；
- 添加 auxiliary loss；
- 改 object tokens；
- 使用 Final Test；
- 启动 real robot；
- 自动进入 Confirmation。

---

## 27. Completion Rule

结果由用户 + ChatGPT 验收：

```text
PASS
→ 冻结 P3
→ C0 Position Offline Confirmation (>=3 seeds + Final Test)

PARTIAL
→ 根据证据决定是否值得一个新的单变量 Discovery

FAIL
→ 停止 fixed-scale 路线
→ 转向新的 spatial representation / fusion intervention
```

本轮结束后停止。
