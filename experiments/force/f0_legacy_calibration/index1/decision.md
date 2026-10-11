# Experiment Decision Contract

> Revision: r0
> 本文件由用户与 ChatGPT 冻结，Codex 只读。

---

## 1. Identity & Mode

- **Experiment ID**: `F0_LEGACY_FORCE_CALIBRATION`
- **Track**: `FORCE`
- **Stage**: `F0_SIGNAL_CALIBRATION`
- **Mode**: `DIAGNOSIS`
- **Data Source**: 旧 LeRobot 数据集
- **Model Training**: 禁止
- **Real-Robot Execution**: 禁止
- **Purpose**: 利用历史采集数据判断 `gripper.load` / `gripper.current` / `gripper.pos` 是否具备可用于 contact / grasp / hold / release 建模的物理信息

### Current Authorization

本轮只授权：

> 对旧数据集中的 `gripper.pos`、`gripper.load`、`gripper.current`、gripper action 及相关时间序列进行离线诊断分析。

本轮明确不授权：

- 训练 Force 模型；
- 修改 ACT；
- 修改 Position 模型；
- 设计新的控制器并真机运行；
- 修改旧数据；
- 删除失败 episode；
- 使用 Final Test 做模型选择。

---

## 2. Research Question

> **RQ:** 旧数据集中记录的 `gripper.load` 与 `gripper.current` 是否能够在时间序列层面区分自由状态、闭合、接触、稳定抓持、悬空保持、重新获得桌面支撑和释放等关键物理阶段？

次要问题：

1. `load` 和 `current` 哪一个更稳定？
2. 单帧绝对值是否足够，还是必须使用时间窗口？
3. 旧数据中的 expert gripper action 在 Hold 阶段是否稳定？
4. 真机曾出现的“Lift 后重新张爪”更可能源于数据本身还是 policy prediction？

---

## 3. Paper Relevance

### Candidate Claim

本轮如果成功，只支持：

> 历史数据中的 gripper proprioceptive signals 含有可利用的 contact / grasp / hold 状态信息，值得进入 Force-aware policy/controller 的后续研究。

### Non-Claims

本轮即使 PASS 也不能证明：

- Force-aware policy 已经有效；
- load/current 能直接代表真实物理力；
- 已实现闭环 force control；
- 真机 Hold 成功率会提升；
- 某一个具体 Force 网络结构最好。

---

## 4. Baseline / Intervention

### Baseline

无模型 baseline。

### Intervention

无模型 intervention。

### Only Intended Scientific Difference

本轮不是模型比较，而是：

> 对旧数据中的 force-related sensor time series 做阶段化诊断。

---

## 5. Frozen Contract

### Data

优先使用旧项目中已存在的数据集。

不得因为某些 episode 不漂亮而选择性删除。

### Required Signals

至少尝试读取：

- `gripper.pos`
- `gripper.load`
- `gripper.current`
- gripper action / target
- 时间索引 / frame index

若存在，也读取：

- arm joint positions
- camera timestamps
- episode success / failure metadata

### Data Status

旧数据集定义为：

`LEGACY_DIAGNOSTIC_EVIDENCE`

不得重新定义为新论文正式训练集。

---

## 6. Phase Definitions

优先识别以下阶段：

### S0 — Free/Open
夹爪张开，无目标接触。

### S1 — Closing
gripper command 正在向闭合方向变化。

### S2 — Contact / Grasp Formation
夹爪继续闭合，但 `pos/load/current` 中出现与自由闭合不同的变化。

### S3 — Stable Grasp
夹爪已建立稳定接触，机械臂尚未或刚准备抬升。

### S4 — Suspended Hold
杯子离开桌面并保持。

### S5 — Support Regained
杯子重新接触桌面，但 gripper 尚未释放。

### S6 — Release
gripper 开始张开并解除接触。

若旧数据无法可靠区分某些阶段，应标记：

`UNRESOLVED_PHASE`

不得强行伪造标签。

---

## 7. Hypotheses

### H1 — Contact Information

> `load` 和/或 `current` 在 free → contact 时存在稳定、可重复变化。

### H2 — Hold Information

> stable grasp / suspended hold 与 free/open 状态在至少一种 force-related signal 上具有可区分结构。

### H3 — Temporal Information

> 若单帧值不足，短时间窗口中的均值、方差、斜率或变化率可以提高阶段可分性。

### H4 — Expert Gripper Stability

> 合格 demonstration 中，Hold 阶段 expert gripper action 应基本维持闭合，而不是系统性重新打开。

---

## 8. Falsification Conditions

### F1
不同阶段的 `load/current` 分布高度重叠，跨 episode 无稳定趋势。

### F2
所谓阶段差异仅来自 `gripper.pos` 几何变化，`load/current` 没有额外信息。

### F3
只有个别 episode 出现明显信号，其余数据不可重复。

### F4
Hold 阶段 expert action 本身频繁重新张开，说明数据语义存在系统性问题。

### F5
信号存在严重饱和、符号混乱、缺失或异常，无法可靠解释。

---

## 9. Mandatory Gates

### G0 — Provenance
确认：
- 使用的是哪一个旧数据集；
- episode 数；
- schema；
- 数据来源；
- 是否存在 NOMASTER / MASTER 混合问题。

### G1 — Signal Availability
确认：
- `pos/load/current/action` 是否真实存在；
- 是否每帧都有；
- 是否时间顺序正确；
- 是否存在大量缺失或异常值。

### G2 — Phase Recoverability
判断：
> 能否根据动作、gripper trajectory 和必要的运动信息恢复主要阶段。

PASS 不要求全部阶段完美自动标注。

### G3 — Single-Episode Signal Structure
至少对代表性 episode 绘制：
- `gripper.pos`
- `gripper.load`
- `gripper.current`
- `gripper action`

随时间变化曲线。

### G4 — Cross-Episode Consistency
跨多个 episode 比较：
- free
- closing
- grasp
- hold
- release

等阶段的分布。

### G5 — Temporal Separability
比较：

#### 单帧特征
- load
- current
- pos

#### 时间窗口特征
- mean
- std
- slope
- delta
- min / max

判断 temporal window 是否明显更有判别力。

### G6 — Hold Action Quality
统计 Hold / Lift 阶段 expert gripper command 是否：
- 稳定闭合；
- 波动；
- 系统性回开。

---

## 10. Primary Metrics

本轮不使用单一模型 accuracy 作为主指标。

主要报告：

### Signal Separation
对关键阶段比较：
- median
- IQR
- P10–P90
- standardized effect size / distribution overlap

### Temporal Consistency
跨 episode 的：
- signal trend consistency
- sign consistency
- contact transition consistency

### Hold Command Stability
例如：
- Hold 阶段 gripper action range
- Hold 阶段 reopen event rate

---

## 11. Required Visualizations

至少生成：

1. 代表性成功 episode 的 `gripper.pos / load / current / action` 时间曲线并标阶段；
2. 多个 episode 的阶段对齐曲线；
3. 不同阶段 `load/current` 分布图；
4. 成功 vs 失败 episode 的 force timeline 对比。

若存在“Lift 后重新张爪”的旧失败样例，应专门加入对应示例。

---

## 12. PASS / PARTIAL / FAIL

### PASS

满足：
- 至少一种 `load/current` 信号在关键阶段存在稳定、跨 episode 可重复结构；
- temporal 或 absolute feature 至少一种具备明显区分能力；
- 数据质量足够支持下一轮 Force Module Discovery。

### PARTIAL

例如：
- `current` 有明显价值，但 `load` 不稳定；
- 单帧不可分，但 temporal window 有价值；
- contact 可识别，但 support-regained 不清晰。

PARTIAL 仍可能允许进入下一轮，但下一轮必须按实际有效信号设计。

### FAIL

例如：
- load/current 基本没有稳定信息；
- 信号严重异常或不可复现；
- 阶段间完全不可分。

此时禁止直接设计复杂 Force Module，应先检查硬件、采样和信号定义。

### BLOCKED

例如：
- 找不到旧数据；
- schema 缺失；
- load/current 实际未记录；
- 数据损坏。

---

## 13. Codex MAY

- 搜索旧数据集；
- 编写只读分析脚本；
- 解析 LeRobot dataset；
- 生成 CSV / JSON；
- 画图；
- 做统计分析；
- 自动/半自动阶段检测；
- 对成功/失败 episode 做分组；
- 创建 `proposal.md`。

---

## 14. Codex MUST

- 保留原始数据不变；
- 报告真实 schema；
- 明确自动推断的阶段和人工确定阶段之间的区别；
- 不把猜测写成 ground truth；
- 报告所有异常；
- 保存分析脚本和 raw evidence；
- 区分 `load`、`current` 和 `pos` 的贡献。

---

## 15. Codex MUST NOT

- 修改 `decision.md`；
- 删除旧数据；
- 重标数据以迎合预期；
- 训练 Force 模型；
- 训练新的 ACT；
- 修改机器人控制；
- 启动真机；
- 因结果不好修改阶段定义；
- 因结果不好选择性排除 episode；
- 自行进入 F1。

---

## 16. Required Deliverables

Codex 必须写入：

```text
experiments/force/f0_legacy_calibration/index1/
├── decision.md
└── result/
    ├── summary.md
    ├── metrics.json
    ├── validation.json
    ├── figures/
    │   ├── representative_timeline.*
    │   ├── aligned_phase_timeline.*
    │   ├── phase_distribution.*
    │   └── hold_action_analysis.*
    └── proposal.md       # optional
```

Raw evidence：

```text
results/experiments/F0_LEGACY_FORCE_CALIBRATION/
```

---

## 17. Completion Rule

完成 F0 后停止。

不得自动训练 Force Module。

下一步只能由用户与 ChatGPT 根据结果决定：

```text
F0 PASS
→ 设计 Force Discovery

F0 PARTIAL
→ 按实际有效信号重新设计最小 Force Module

F0 FAIL
→ 优先修正数据采集 / signal calibration

F0 BLOCKED
→ 解决数据或 schema 问题
```
