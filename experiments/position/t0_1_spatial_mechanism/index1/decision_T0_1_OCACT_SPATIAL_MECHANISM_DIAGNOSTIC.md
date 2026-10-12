# Experiment Decision Contract

> Revision: r0
> 本文件由用户与 ChatGPT 冻结，Codex 只读。

# T0.1 — OC-ACT Spatial Mechanism Diagnostic

## 1. Identity

- **Experiment ID**: `T0_1_OCACT_SPATIAL_MECHANISM_DIAGNOSTIC`
- **Track**: `POSITION`
- **Stage**: `DISCOVERY_DIAGNOSTIC`
- **Mode**: `DIAGNOSIS`
- **Upstream**: `T0_POSITION_DISCOVERY_ACT_VS_OCACT = PARTIAL`
- **Policy Retraining**: 禁止
- **Dev Use**: 允许
- **Final Test Use**: 禁止
- **Real Robot**: 禁止

本轮只回答一个问题：

> 当前 P1 OC-ACT 的性能提升究竟是否使用了“目标的空间信息”，以及空间信息主要来自显式 `(x,y)`、local-token 身份/语义，还是仅仅来自 object presence / semantics。

本轮不改变任何模型参数。

---

# 2. Frozen Checkpoints

固定使用 T0 已选出的 checkpoint：

```text
B0:
B0_s1 / 040000

P1:
P1_ocfix2_s1 / 100000
```

不得重新选 checkpoint。

已知 Dev 结果：

```text
B0 inference L1 = 0.241336
P1 inference L1 = 0.230687
relative improvement ≈ 4.43%
```

已知 P1 mechanism：

```text
gate alpha ≈ -0.02939
normal → zero object action delta ≈ 0.07452
normal → shuffled object delta ≈ 0.02036
normal → x-coordinate sign flip delta ≈ 0.00103
```

---

# 3. Frozen Data

只使用 D1 冻结的 12-episode Dev split。

禁止读取：

```text
Final Test
```

不允许根据本轮诊断修改 Dev split。

---

# 4. Frozen Architecture and Artifacts

全部保持不变：

```text
DINOv2-S frozen
4-prototype bank
tau = 0.20
1 global + 4 local object tokens
adapter
pre-encoder cross-attention
gate
ACT backbone
8D NOMASTER state
```

禁止重新训练或修改 architecture。

---

# 5. Why T0 Spatial Swap Was Insufficient

当前 object token 不是只有 `(x,y)`：

```text
semantic feature
+
x
+
y
+
confidence
```

且 local tokens 固定具有：

```text
upper-left
upper-right
lower-left
lower-right
```

的槽位身份。

因此仅做：

```text
x -> -x
```

可能只破坏“显式坐标”，但仍保留：

- local-token slot identity；
- local semantic content；
- object appearance distribution；
- global token；
- 原 ACT Top visual branch 中的目标位置。

所以 T0 的小 spatial-swap effect：

> 不能直接等价于“OC-ACT 没使用空间信息”。

---

# 6. Diagnostic Principle

所有 counterfactual 必须：

> 固定同一模型、同一 input frame、同一 robot state，只改变 object-conditioning representation。

这样才能隔离 object branch 的机制。

主要输出：

```text
Δaction = |a_counterfactual - a_normal|
```

同时报告：

- all-action mean；
- arm-only；
- gripper-only；
- h0 / first-step；
- relevant position-sensitive action dimensions。

---

# 7. Required Counterfactuals

## D0 — Normal

正确 object tokens。

作为 reference。

---

## D1 — Zero All Object Tokens

```text
all 5 tokens = 0
```

用于复现 T0 object-path dependence。

---

## D2 — Shuffle Whole Object Representation

在 batch / matched frames 间交换整组 5 tokens。

保持 T0 定义或给出完全一致复现。

---

## D3 — XY Zero

保留：

```text
semantic
confidence
local slot
```

但：

```text
x = 0
y = 0
```

目的：

> 测显式坐标是否提供额外信息。

---

## D4 — XY Shuffle

在不同 samples 间只打乱：

```text
x,y
```

其余 semantic/confidence/token slots 保持当前 sample。

目的：

> 比 `x→-x` 更彻底破坏显式空间坐标与当前图像的对应关系。

---

## D5 — Left/Right Local Token Swap

交换完整 local token：

```text
UL <-> UR
LL <-> LR
```

必须交换整个 token：

```text
semantic + x + y + confidence
```

global token 保持不变。

目的：

> 破坏左右局部空间结构，而不仅仅翻一个坐标字段。

---

## D6 — Up/Down Local Token Swap

交换：

```text
UL <-> LL
UR <-> LR
```

global token 保持不变。

目的：

> 检查纵向 local structure。

---

## D7 — Local Token Full Permutation

随机排列 4 个 local tokens 的槽位：

```text
T1..T4
```

global token不变。

必须固定 permutation seed。

目的：

> 检查 policy 是否利用 local-token layout。

---

## D8 — Global Only

保留：

```text
T0 global
```

将：

```text
T1-T4 = 0
```

回答：

> 改善是否主要来自“有一个杯子 object embedding”而非局部空间信息。

---

## D9 — Local Only

```text
T0 = 0
T1-T4 保留
```

回答：

> local object structure 是否真正贡献。

---

# 8. Optional Strong Diagnostic — Token Transplant

若实现成本合理，必须优先做。

在 Dev 中构造位置差异明显的 frame pairs：

```text
left ↔ right
near ↔ far
```

对同一个 base observation：

```text
Top/Wrist RGB + state 保持原样
```

只将 object token group 替换成另一帧的 object tokens：

```text
object-token transplant
```

观察 action 是否向 donor target position 对应方向发生变化。

该实验用于回答：

> 在 ACT 主视觉输入完全不变时，仅改变 object representation 能否推动动作产生方向性变化。

---

# 9. Pair Selection for Token Transplant

只使用 Dev。

位置来源必须使用已有可信 cup-position estimator / P0 object center。

构造：

```text
>= 20 left-right pairs
>= 20 near-far pairs
```

若 Dev 数据不足以满足数量：

> 使用全部可构造的高置信 pair，并报告数量，不得访问 Final Test。

Pair 需尽量匹配：

- 相近任务阶段；
- 相近 robot state；
- 避免把 Reach frame 和 Release frame 配对。

---

# 10. Directional Mechanism Metrics

不仅看：

```text
absolute action delta
```

还必须检查：

> action change 的方向是否与 donor target 的位置变化方向一致。

例如 donor 从 left→right：

```text
Δtarget_x > 0
```

检查与横向 reach 最相关的 action dimension：

```text
sign(Δaction_relevant)
```

是否具有稳定方向性。

如果关节空间中“哪个维度代表左右”不是固定单一维度：

> 必须先从 demonstration / kinematics / existing action-position correlation 中确定 diagnostic mapping，不得看结果后挑最有利维度。

---

# 11. Action-Position Correlation Baseline

在 Dev expert actions 上先计算：

```text
cup x / y
vs
h0 action dimensions
```

用于预先识别：

- horizontal-position-sensitive dimensions；
- longitudinal-position-sensitive dimensions。

选择规则必须在查看 counterfactual result 前确定。

---

# 12. Primary Mechanism Questions

必须分别回答：

### Q1
Object branch 是否被使用？

证据：

```text
D1 zero
D2 shuffle
```

### Q2
显式 x/y 是否被使用？

证据：

```text
D3 XY zero
D4 XY shuffle
```

### Q3
Local spatial layout 是否被使用？

证据：

```text
D5 LR swap
D6 UD swap
D7 permutation
```

### Q4
Global object semantics 与 local spatial structure 谁贡献更多？

证据：

```text
D8 global-only
D9 local-only
```

### Q5
Object token 单独是否能产生方向性的 action steering？

证据：

```text
token transplant
```

---

# 13. PASS Criteria

本轮不要求每个 counterfactual 都必须巨大。

### PASS

必须满足：

1. T0 的 object dependence 可复现；
2. 至少一种更完整的 spatial intervention：
   - LR/UD local swap，
   - local permutation，
   - XY shuffle，
   - token transplant
   对 arm action 产生稳定可测变化；
3. 变化与目标位置方向至少在一个主要轴上具有合理 directional consistency；
4. 机制不能仅由 gripper action 驱动；
5. Dev-only，无 Final Test contamination。

如果 token transplant 可执行：

> directional transplant evidence 优先级最高。

### PARTIAL

例如：

- object branch明显有用；
- local/global ablation有效；
- 但方向性 steering 仍弱或只有 x 轴成立。

### FAIL

例如：

- 除 zero object 外所有空间干预都几乎无效；
- token transplant 不产生方向性变化；
- P1 只利用“object present / appearance”而没有可证明的 target-location mechanism。

FAIL 不代表 P1 Dev L1 改善是假的，只表示：

> 当前“object-conditioned position mechanism”论文解释不成立。

---

# 14. No Retraining

本轮严禁：

- 修改 gate；
- 放大 x/y；
- 改 adapter；
- 改 token order；
- 改 architecture；
- 再训练 100k；
- fine-tune。

如果诊断发现机制不足：

> 结束为 FAIL/PARTIAL，下一轮另开新 scientific intervention。

---

# 15. Required Outputs

```text
experiments/position/t0_1_spatial_mechanism/index1/
├── decision.md
└── result/
    ├── summary.md
    ├── validation.json
    ├── metrics.json
    ├── intervention_results.json
    ├── action_position_mapping.json
    ├── transplant_pairs.json
    ├── figures/
    │   ├── intervention_action_delta.*
    │   ├── global_vs_local.*
    │   ├── xy_ablation.*
    │   ├── local_swap.*
    │   ├── transplant_directionality.*
    │   └── position_action_correlation.*
    └── proposal.md
```

---

# 16. Codex MAY

- 编写离线 counterfactual evaluation；
- 读取 B0/P1 固定 checkpoints；
- 使用 Dev；
- 构造固定 token permutation；
- 构造 Dev token-transplant pairs；
- 做统计和可视化；
- 创建 proposal.md。

---

# 17. Codex MUST

- 不重新训练；
- 不改 checkpoints；
- 不读 Final Test；
- 不看结果后挑 action dimension；
- 明确区分 explicit xy 与 local-slot spatial info；
- 报告 arm 与 gripper 分量；
- 保存固定 permutation/pair manifest；
- 完成后停止。

---

# 18. Codex MUST NOT

- 修改本 decision.md；
- 使用 Final Test；
- 启动 real robot；
- 修改 model architecture；
- 放大 coordinate scale；
- 改 gate；
- 重建 prototype；
- 调 tau；
- 根据诊断结果立刻重训；
- 自动进入 Confirmation。

---

# 19. Completion Rule

结果由用户 + ChatGPT 验收：

```text
PASS
→ 冻结当前 OC-ACT
→ C0 Position Offline Confirmation (>=3 seeds + Final Test)

PARTIAL
→ 决定是否需要一个新的、单变量 architecture intervention

FAIL
→ 不以当前结构主张 spatial object-conditioning mechanism
```

本轮结束后停止。
