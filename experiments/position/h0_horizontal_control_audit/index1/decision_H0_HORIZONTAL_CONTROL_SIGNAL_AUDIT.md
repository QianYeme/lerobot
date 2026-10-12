# Experiment Decision Contract

> Revision: r0
> 本文件由用户与 ChatGPT 冻结，Codex 只读。

# H0 — Horizontal Control Signal Audit

## 1. Identity

- **Experiment ID**: `H0_HORIZONTAL_CONTROL_SIGNAL_AUDIT`
- **Track**: `POSITION`
- **Stage**: `MECHANISM_DIAGNOSTIC`
- **Mode**: `DIAGNOSIS`
- **Policy Training**: 禁止
- **Architecture Change**: 禁止
- **Final Test**: 禁止
- **Real Robot**: 禁止

本轮不是继续搜索网络结构。

本轮只回答：

> 我们此前将“桌面横向目标变化”简化为 `h0 action dim0` 的方向一致性，这个诊断映射是否真的正确？  
> 对 SO100/SO101 当前姿态和演示轨迹而言，杯子横向位移究竟由哪些关节、哪个时间尺度、哪种联合 action direction 实现？

---

# 2. Motivation

目前已经出现跨架构一致现象：

```text
P1 OC-ACT
T3 Joint State-Spatial
R2 Action-Level Spatial
R1 Relational Spatial
R3 FiLM
```

都没有通过既有：

```text
horizontal primary action dim = dim0
```

的 steering gate。

但 longitudinal 多次通过。

同时 N0 中：

```text
R2 normal Dev L1 ≈ 0.135205
R2 spatial-zero Dev L1 ≈ 0.244327
```

说明 R2 spatial path 对 policy 输出非常重要；

然而：

```text
x transplant / spatial shuffle
```

对总体输出变化极小。

此外，当前评估配置：

```text
chunk_size = 100
n_action_steps = 1
```

而既有 steering 诊断主要观察：

```text
h0 / first action
```

因此存在三个必须排除的可能性：

1. 横向末端运动并不主要对应单一 `action dim0`；
2. 横向 steering 是多关节联合方向；
3. target-position effect 可能在 action chunk 的后续时刻才明显，而不是 h0。

在继续 architecture search 前必须先验证诊断本身。

---

# 3. Frozen Data Boundary

允许：

```text
Train 36
Dev 12
```

禁止：

```text
Final Test 12
```

Train 用于：

```text
identify / fit diagnostic mapping
```

Dev 只用于：

```text
independent validation of frozen mapping
```

不得使用 Dev 反向调整 mapping。

---

# 4. Frozen Target Position

cup position 使用现有冻结：

```text
x_global
y_global
confidence_global
```

来自：

```text
P0.1
4 prototypes
tau = 0.20
```

禁止：

- GT detector
- Final Test annotation
- 手工重新定义位置
- 看结果后改 x/y estimator

---

# 5. Phase Scope

本轮只分析 Position/reach 相关阶段。

优先：

```text
Reach
Fine Alignment
Pre-Grasp
```

如果没有显式 phase label：

使用预注册规则构造 reach window。

推荐规则：

1. episode 前段；
2. gripper 尚未进入稳定闭合；
3. 排除 grasp / hold / lower / release；
4. 使用 action/state dynamics 辅助识别。

必须保存：

```text
phase_selection_manifest.json
```

禁止看 correlation 结果后再改 phase 规则。

---

# 6. Audit A — Expert Joint-Space Signal

在 Train reach frames 上计算：

```text
cup x
vs
each action dimension
```

不仅计算 h0。

必须对以下时间尺度分别分析：

```text
h0
mean action over first 3 steps
mean action over first 5 steps
mean action over first 10 steps
cumulative joint delta over first 3/5/10 steps
```

若当前 policy action chunk 数据可访问更长 horizon：

> 使用真实 chunk，不得人为生成未来动作。

报告每个 action dim：

- Pearson
- Spearman
- sign
- effect size
- bootstrap CI

同样对 cup y 做纵向参考。

---

# 7. Audit B — Multi-Joint Horizontal Direction

禁止只假设单一 joint。

在 **Train only** 上寻找一个冻结线性 projection：

```text
u_horizontal = w_x^T * Δq
```

其中：

```text
Δq = arm joint action / short-horizon joint delta
```

目标：

> 找到最能表示 expert 横向 target response 的多关节方向。

允许方法二选一，优先使用 A：

### Method A — Robot Kinematics / Jacobian

如果项目中有可靠的：

- SO100/SO101 kinematics
- URDF
- calibrated FK/Jacobian

则用当前 state 计算：

```text
joint delta
→ predicted EE Cartesian delta
```

冻结桌面横向轴：

```text
ΔEE_horizontal
```

作为 primary metric。

### Method B — Train-Only Empirical Projection

若没有可靠 FK/Jacobian：

使用 Train reach frames 拟合简单、可解释的：

```text
linear regression / ridge
```

从 arm joint delta 预测：

```text
cup target x-related expert response
```

只允许线性模型。

禁止：

- neural network
- nonlinear search
- Dev tuning

regularization 若需要，只能 Train 内部固定/交叉验证。

---

# 8. Preferred Primary Metric

如果 Method A 可用：

```text
PRIMARY_HORIZONTAL_METRIC
= predicted EE horizontal displacement
```

这是优先方案。

如果只能 Method B：

```text
PRIMARY_HORIZONTAL_METRIC
= frozen train-only multi-joint projection
```

无论使用哪种方法：

> 必须在查看候选模型 counterfactual 之前冻结。

---

# 9. Coordinate/Sign Calibration

必须明确：

```text
image x increasing direction
table transverse physical direction
robot Cartesian horizontal direction
```

三者之间的 sign。

保存：

```text
coordinate_convention.json
```

防止：

```text
image left/right
```

与机器人 world/base frame 左右符号相反。

---

# 10. Dev Validation of Mapping

mapping 冻结后在 Dev expert demonstrations 上验证：

### Required

```text
target Δx
vs
expert projected horizontal action
```

要求报告：

- Pearson
- Spearman
- sign accuracy
- effect size
- bootstrap CI
- left/right group mean difference

如果 frozen mapping 在 Dev expert 上都不能稳定表示横向控制：

> 之前的 horizontal mechanism gate 判定为 INVALID / UNRESOLVED。

此时禁止继续用它否定模型。

---

# 11. Temporal Audit

必须回答：

> 横向响应最早在哪个时间尺度出现？

固定比较：

```text
1 step
3 steps
5 steps
10 steps
```

报告每个 horizon 的：

```text
target-x ↔ projected action correlation
direction consistency
signal magnitude
```

如果 h0 弱、5/10-step 强：

> 后续模型 steering 评估必须使用冻结的短 horizon，而不是只看 h0。

---

# 12. State-Matched Expert Pair Audit

在 Train 上构造：

```text
similar robot state
similar phase
different cup x
```

的 expert frame pairs。

要求：

```text
>= 30 pairs if available
```

比较 expert action / projected EE motion：

```text
left target vs right target
```

目的：

> 排除 cup x 与机器人当前 state 混杂造成的伪 correlation。

Dev 上使用同一冻结匹配规则独立验证。

---

# 13. Re-Evaluate Existing Models — No Retraining

只有在 horizontal mapping 完全冻结以后，才允许重新评估已有 checkpoints：

```text
P1 original OC-ACT
T3/P4
R2 Action-Level
R1 Relational
R3 FiLM
```

如 checkpoint 仍可用。

禁止重新训练。

对每个模型重复已有 transplant：

```text
large horizontal donor transplant
same-position control
random control
```

但 primary outcome 改为：

```text
frozen horizontal projection / EE horizontal displacement
```

而不是强制 `action dim0`。

---

# 14. Model Re-Evaluation Metrics

每个模型报告：

```text
direction consistency
Pearson(Δtarget_x, Δprojected_horizontal)
Spearman
mean |Δprojected_horizontal|
large-position / same-position response ratio
```

同时保留：

```text
dim0
```

作为 secondary historical metric，便于和旧结果对应。

---

# 15. Chunk-Horizon Re-Evaluation

如果训练模型可以输出完整 action chunk：

必须同时评估：

```text
h0
first 3
first 5
first 10
```

的 projected horizontal response。

不得因为 `n_action_steps=1` 的部署设置而只分析第一步。

如果当前 checkpoint API 只暴露一帧 action，但内部 decoder 有 chunk：

> 允许离线读取 raw predicted chunk，仅用于分析，不改变部署 policy。

---

# 16. PASS / FAIL Interpretation

本轮不是模型 PASS，而是诊断 PASS。

## H0-A — DIAGNOSTIC_VALID

如果：

1. Train 找到稳定 horizontal mapping；
2. frozen mapping 在 Dev expert 上独立验证成立；
3. coordinate sign 明确；
4. temporal horizon 明确。

则：

```text
HORIZONTAL_DIAGNOSTIC_READY = TRUE
```

后续所有模型统一使用新冻结 metric。

## H0-B — OLD_METRIC_INVALID

如果：

```text
dim0/h0 weak
```

但：

```text
multi-joint / Cartesian / multi-step mapping strong
```

则明确记录：

> 过去以 dim0/h0 为主的 horizontal FAIL 不能作为最终机制结论，需要按新 metric 重验。

## H0-C — DATA_SIGNAL_WEAK

如果即使 expert demonstration：

```text
state-matched target x
```

也不能稳定对应可测横向动作：

则问题不优先归因于模型。

记录：

```text
HORIZONTAL_SUPERVISION_WEAK = TRUE
```

下一步应回到数据/任务设计，而不是继续 architecture search。

---

# 17. Decision After Re-Evaluation

### Case 1

若某已有模型在新冻结 horizontal metric 下：

```text
horizontal >= 0.70
longitudinal preserved
```

则：

> 不再训练新架构；
> 选择该模型进入下一轮 Confirmation 设计。

### Case 2

若 expert horizontal signal 强，但所有模型仍 FAIL：

> 才授权新的 relative target-state architecture。

### Case 3

若 expert horizontal supervision 本身弱：

> 停止 architecture search；
> 重新评估数据采集和 position supervision。

---

# 18. Required Outputs

```text
experiments/position/h0_horizontal_control_audit/index1/
├── decision.md
└── result/
    ├── summary.md
    ├── validation.json
    ├── phase_selection_manifest.json
    ├── coordinate_convention.json
    ├── expert_joint_signal.json
    ├── expert_temporal_signal.json
    ├── horizontal_mapping.json
    ├── state_matched_pairs.json
    ├── dev_mapping_validation.json
    ├── model_re_evaluation.json
    ├── figures/
    │   ├── cup_x_vs_joint_actions.*
    │   ├── cup_x_vs_projected_horizontal.*
    │   ├── horizon_signal_strength.*
    │   ├── state_matched_expert_pairs.*
    │   └── model_horizontal_re_evaluation.*
    └── proposal.md
```

---

# 19. Codex MAY

- 读取 Train/Dev demonstrations；
- 使用已有 robot FK/Jacobian/URDF；
- 做 Train-only linear projection；
- 读取现有 checkpoints；
- 输出 raw predicted action chunks；
- 重新运行离线 transplant；
- 做统计与图表；
- 创建 proposal.md。

---

# 20. Codex MUST

- 不训练任何 policy；
- 不修改 architecture；
- mapping 只能在 Train 构造；
- Dev 只能验证 frozen mapping；
- 不读 Final Test；
- 明确 image/table/robot frame sign；
- 对多个 horizon 做审计；
- 优先使用 FK/Cartesian metric；
- 完成后停止。

---

# 21. Codex MUST NOT

- 修改本 decision.md；
- 使用 Final Test；
- 重训 P1/T3/R1/R2/R3；
- 根据候选模型结果调整 horizontal mapping；
- 看 Dev 后重新选 projection；
- 再新增 architecture candidate；
- 启动 real robot；
- 自动进入 Confirmation。

---

# 22. Completion Rule

本轮完成后必须明确返回以下三者之一：

```text
A. OLD_HORIZONTAL_METRIC_INVALID
B. HORIZONTAL_METRIC_VALID_AND_MODELS_REEVALUATED
C. HORIZONTAL_SUPERVISION_WEAK
```

然后停止。

由用户 + ChatGPT 决定：

```text
existing candidate → Confirmation
or
relative target-state architecture
or
data redesign
```
