# Experiment Decision Contract

> Revision: r0
> 本文件由用户与 ChatGPT 冻结，Codex 只读。

# T0.2 — OC-ACT Token Transplant Directionality

## 1. Identity

- **Experiment ID**: `T0_2_OCACT_TOKEN_TRANSPLANT_DIRECTIONALITY`
- **Track**: `POSITION`
- **Stage**: `DISCOVERY_DIAGNOSTIC`
- **Mode**: `DIAGNOSIS`
- **Upstream**:
  - `T0_POSITION_DISCOVERY_ACT_VS_OCACT = PARTIAL`
  - `T0_1_OCACT_SPATIAL_MECHANISM_DIAGNOSTIC = PARTIAL`
- **Policy Retraining**: 禁止
- **Architecture Change**: 禁止
- **Dev Use**: 允许
- **Final Test Use**: 禁止
- **Real Robot**: 禁止

本轮只回答：

> 在 Top/Wrist RGB 和 robot state 都保持不变时，仅替换 object token representation，是否能够以与 donor cup position 一致的方向改变 ACT action。

这是当前 OC-ACT 是否具有“object-conditioned spatial steering”机制的最后一个关键 Discovery 诊断。

---

# 2. Frozen Checkpoint

固定：

```text
P1_ocfix2_s1 / 100000
```

不得重新选 checkpoint。

不得使用 B0 重新训练。

---

# 3. Frozen Model Components

全部保持不变：

```text
Frozen DINOv2-S
4 prototypes
tau = 0.20
1 global + 4 local tokens
adapter
pre-encoder cross-attention
gate
ACT backbone
8D NOMASTER
```

禁止：

- 修改 coordinate scale；
- 修改 gate；
- 修改 local-token order；
- 修改 prototype；
- 修改 pooling；
- fine-tune；
- 再训练。

---

# 4. Data Boundary

只使用：

```text
Dev 12 episodes
```

禁止读取：

```text
Final Test
```

若 Dev 不足以构造理想 pair：

> 使用所有可构造高置信 pair，不得访问 Final Test 补数量。

---

# 5. Coordinate Source

cup position 必须使用：

```text
P1 raw global object token x/y
```

即 T0.1 已使用的：

```text
indices 384 / 385
```

或其等价已冻结实现。

不得使用看完结果后新挑的 position estimator。

---

# 6. Frame Eligibility

只从具有高 object confidence 的 Dev frames 中选 pair。

必须排除：

- object confidence 异常低；
- token 非 finite；
- 明显 release / retreat 等与 Reach 无关阶段；
- phase 极不匹配的 frame 对。

优先使用：

```text
Reach
Fine Alignment
Pre-Grasp
```

阶段。

如果 phase label 不是显式 ground truth：

> 使用冻结规则从 action/state/episode timing 近似筛选，并记录算法。

---

# 7. State Matching

Transplant pair 不能只按 cup position 差异随机配对。

对于 base frame 与 donor frame，应尽量满足：

```text
robot state similar
task phase similar
gripper state similar
```

建议以 8D state 的标准化距离做 nearest-neighbor matching。

匹配规则必须在看 transplant result 前冻结。

---

# 8. Pair Types

构造两类 pair：

## 8.1 Horizontal Pairs

```text
left donor ↔ right donor
```

要求 donor cup x 差异达到 Dev 内显著分位差。

优先：

```text
|Δx| >= Dev x-span 的 50%
```

若数量不足，可降到 40%，但必须在执行前固定。

目标：

```text
>= 20 pairs
```

不足则使用全部符合规则的 pairs，并报告数量。

---

## 8.2 Longitudinal Pairs

```text
near donor ↔ far donor
```

要求：

```text
|Δy| >= Dev y-span 的 50%
```

目标：

```text
>= 20 pairs
```

同样不足则使用全部。

---

# 9. Transplant Definition

对 base frame：

```text
Top RGB = base
Wrist RGB = base
robot state = base
```

保持完全不变。

正常 forward：

```text
a_normal
```

Transplant forward：

```text
object tokens = donor object tokens
```

其余输入仍然是 base：

```text
a_transplant
```

定义：

```text
Δa = a_transplant - a_normal
```

---

# 10. Required Transplant Variants

至少做以下三种：

## V1 — Whole Object Token Transplant

替换：

```text
T0-T4 全部
```

目的：

> 测完整 object representation steering。

---

## V2 — Local-Only Transplant

替换：

```text
T1-T4
```

保留 base：

```text
T0 global
```

目的：

> 测 local spatial layout 单独 steering。

---

## V3 — Global-Only Transplant

替换：

```text
T0 global
```

保留 base：

```text
T1-T4 local
```

目的：

> 测 global semantic/position token steering。

---

# 11. Pre-Registered Action Mapping

禁止看完 transplant 后挑“最好看的 action dimension”。

使用 T0.1 已冻结的 expert action-position correlation。

## Horizontal axis

基于 expert x correlation，固定主要候选 action dimensions：

```text
dim0
dim2
```

其中绝对相关最高优先：

```text
dim0
```

必须报告所有 6 dims，同时以 dim0 为 horizontal primary diagnostic。

## Longitudinal axis

基于 expert y correlation，固定主要候选：

```text
dim2
dim1
dim3
```

以绝对相关最高：

```text
dim2
```

为 longitudinal primary diagnostic。

不得执行后改 primary dimension。

---

# 12. Directional Consistency

对于每个 transplant pair：

## Horizontal

若 donor 相对 base：

```text
Δx_target > 0
```

则检查：

```text
sign(Δa_dim0)
```

是否与 expert correlation 所预测方向一致。

若 expert corr(dim0, x) < 0：

```text
Δx_target > 0
→ 预期 Δa_dim0 < 0
```

反之亦然。

## Longitudinal

同理使用：

```text
dim2
```

和 expert corr(dim2, y)。

---

# 13. Primary Metrics

分别对：

```text
V1 whole
V2 local-only
V3 global-only
```

报告：

### 13.1 Direction Accuracy

```text
horizontal directional consistency rate
longitudinal directional consistency rate
```

### 13.2 Signed Response Strength

```text
corr(Δtarget_x, Δaction_dim0)
corr(Δtarget_y, Δaction_dim2)
```

报告：

- Pearson；
- Spearman。

### 13.3 Action Magnitude

报告：

```text
mean |Δaction|
h0 arm |Δ|
per-dim |Δ|
```

### 13.4 Axis Selectivity

检查 horizontal transplant 是否主要影响 horizontal-sensitive dims，而不是随机全部维度等量波动。

纵向同理。

---

# 14. Null / Control Tests

必须加入：

## C1 — Same-Position Token Swap

选择 cup position 很接近的 donor：

```text
|Δx|, |Δy| 小
```

替换 whole object tokens。

预期：

> action change 应明显小于大位置差 transplant。

## C2 — Random Unmatched Token Swap

随机 donor，不匹配 state/phase。

只作为上界/噪声参考，不作为 mechanism 正证据。

---

# 15. PASS Criteria

### PASS

至少满足：

1. Whole-token transplant 的方向性在 horizontal 或 longitudinal 至少一个轴上稳定成立；
2. 对应 primary action dim：
   - direction consistency 明显高于 chance；
   - correlation sign 与 expert mapping 一致；
3. large-position transplant 的 action response 明显大于 same-position control；
4. local-only 或 whole-token 至少一个产生稳定 steering；
5. 结果不是 gripper-only；
6. Dev-only，无 Final Test contamination。

建议但不强制的参考标准：

```text
direction consistency >= 0.70
```

并具有与 expert mapping 一致的相关性。

---

### PARTIAL

例如：

- 只有 y 轴方向性成立；
- whole-token 有 steering，但 local-only 较弱；
- direction consistency 在 0.60–0.70；
- action response存在但 axis selectivity一般。

---

### FAIL

例如：

- large-position 与 same-position transplant 差异很小；
- direction consistency 接近随机；
- action 变化虽大但方向与 donor cup position 无关；
- 仅 gripper action 变化明显。

---

# 16. Interpretation Rule

如果 PASS：

> 当前 OC-ACT 可以主张“object representation 对动作具有因果空间 steering 作用”，即使显式 numeric x/y 字段本身贡献很弱。

此时空间信息主要可归因于：

```text
local token layout
+
object semantic/spatial structure
```

而不是显式 `(x,y)` 数值。

如果 FAIL：

> 当前结构的 Dev 改善仍可保留，但不能主张明确的 object-conditioned spatial steering mechanism。

然后才考虑新的单变量 architecture intervention。

---

# 17. Required Outputs

```text
experiments/position/t0_2_token_transplant/index1/
├── decision.md
└── result/
    ├── summary.md
    ├── validation.json
    ├── metrics.json
    ├── pair_manifest.json
    ├── transplant_results.json
    ├── controls.json
    ├── figures/
    │   ├── horizontal_directionality.*
    │   ├── longitudinal_directionality.*
    │   ├── whole_vs_local_vs_global.*
    │   ├── large_vs_same_position.*
    │   └── per_action_dimension.*
    └── proposal.md
```

---

# 18. Codex MAY

- 读取固定 P1 checkpoint；
- 使用 Dev；
- 构造 matched frame pairs；
- 执行 token transplant；
- 执行 control swaps；
- 做统计和图表；
- 创建 proposal.md。

---

# 19. Codex MUST

- 不重训；
- 不改模型；
- 不读 Final Test；
- 固定 pair 选择规则；
- 固定 action primary dims；
- 报告 horizontal / longitudinal 分开结果；
- 报告 arm / gripper 分开结果；
- 完成后停止。

---

# 20. Codex MUST NOT

- 修改本 decision.md；
- 重新训练；
- 修改 coordinate scale；
- 修改 gate；
- 修改 prototype；
- 调 tau；
- 看结果后换 action dimension；
- 使用 Final Test；
- 启动 real robot；
- 自动进入 Confirmation。

---

# 21. Completion Rule

结果由用户 + ChatGPT 验收：

```text
PASS
→ 冻结当前 OC-ACT
→ C0 Position Offline Confirmation (>=3 seeds + Final Test)

PARTIAL
→ 讨论是否足以进入 Confirmation，或需要单变量新 intervention

FAIL
→ 当前结构不再主张 spatial steering mechanism
→ 新开 architecture intervention
```

本轮结束后停止。
