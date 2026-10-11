# LeRobot 半满透明塑料杯抓取：多轮实验与科研决策工作流

> 目标：把本项目做成一个可持续迭代、可审计、可回退、面向 CCF-A 论文的机器人学习研究过程。
>
> 用户只参与研究大方向、结果理解与最终拍板；ChatGPT 主导科研问题拆解、实验设计、验收和红队审计；Codex 负责代码、训练、评估、诊断、数据整理和工程执行。
>
> 本文件只保存**长期稳定规则**。当前做到哪里写入 `workflow/STATUS.md`；每轮具体科研合同写入 `experiments/<track>/<stage>/indexN/decision.md`；历史事实写入 `docc/日志/工作日志.md`。

---

## 1. 项目目标与研究边界

项目背景：

> 基于 LeRobot，研究半满透明塑料杯的稳定抓取与操作，使机械臂能够完成 Reach → Grasp → Lift → Hold → Place → Release，并尽量避免掉杯和明显洒水。

论文导向：**CCF-A**。

当前长期研究逻辑：

```text
Position / Visual Alignment
        ↓
Reliable Contact
        ↓
Force / Contact-aware Grasp
        ↓
Lift / Hold / Place / Release
        ↓
System Integration
```

Position 负责“抓哪里、如何对准”；Force 负责“接触后如何建立和保持稳定夹持”。

最终系统不预先限定必须是：

- 单一神经网络；
- 多阶段策略；
- policy + controller；
- hybrid policy。

以最终效果、稳定性、可归因性和论文证据完整性综合决定。

---

## 2. 三方职责

### 2.1 用户：Project Owner / Final Authority

用户负责：

- 判断研究大方向；
- 判断论文主线是否符合目标；
- 对新研究结构、新主变量、高成本实验和真机正式实验进行最终批准；
- 看完 ChatGPT 的验收与推荐后，决定是否进入下一轮；
- 通过明确回复“继续”表示认可当前结论和下一步方向。

用户不需要参与：

- tensor shape；
- 代码结构；
- CLI 拼接；
- 普通 debug；
- 缓存；
- 日志解析；
- 一般工程实现细节。

### 2.2 ChatGPT：Scientific Lead + Planner + Red Team

ChatGPT 负责：

- 理解项目背景和历史证据；
- 和用户讨论研究问题；
- 每轮只选择一个主要科研不确定性；
- 设计最小充分实验；
- 设计 falsification conditions；
- 冻结 baseline、intervention、数据、指标、Gate 和 PASS/FAIL；
- 与用户共同生成每轮 `decision.md`；
- 验收 Codex 的 `result/`；
- 执行 Builder Review；
- 执行 Red-Team Review；
- 判断当前证据最多支持到哪一级 claim；
- 用通俗中文向用户解释结果；
- 给出下一步明确推荐；
- 在用户回复“继续”前，不生成下一轮正式 `decision.md`。

ChatGPT 不应把微观工程选择反复抛给用户。

### 2.3 Codex：Research Engineer / Executor

Codex 负责：

- 阅读本工作流、`workflow/STATUS.md` 和当前 `decision.md`；
- 把已经冻结的科研语义实现成代码；
- 自主完成正常编程、调试、依赖、性能和数据处理工作；
- 执行训练、离线评估、诊断和已授权真机辅助工作；
- 保存 raw evidence；
- 生成紧凑 `result/`；
- 若发现更好的研究结构，只能生成 `proposal.md`。

核心原则：

> **Codex 可以自主改变“怎么实现”，不能自主改变“研究什么、比较什么、什么算成功”。**

---

## 3. 双循环机制

### 3.1 内层工程循环

同一轮 `decision.md` 内，Codex 可自行反复：

```text
实现 → 运行 → 工程错误 → 调试 → 重跑 → 自动测试 → 收集证据
```

只要科研语义未改变，不需要用户逐项批准。

### 3.2 外层科研循环

固定为：

```text
用户 + ChatGPT 讨论
        ↓
ChatGPT 生成 decision.md
        ↓
用户批准
        ↓
Codex 执行
        ↓
Codex 写 result/ + raw evidence
        ↓
ChatGPT Builder Review
        ↓
ChatGPT Red-Team Review
        ↓
ChatGPT 向用户解释 + 推荐下一步
        ↓
用户讨论 / 修正
        ↓
用户回复“继续”
        ↓
ChatGPT 才生成下一轮 decision.md
```

**没有新的 `decision.md`，Codex 不得进入下一科研轮次。**

---

## 4. `decision.md`：每轮唯一科研合同

路径：

```text
experiments/<track>/<stage>/indexN/decision.md
```

`decision.md` 由 **用户 + ChatGPT** 冻结。

Codex 对其完全只读。

每轮 Core Contract 至少包含：

1. Identity & Mode
2. Research Question
3. Paper Relevance
4. Baseline / Intervention
5. Frozen Contract
6. Hypothesis & Falsification
7. Gates & Metrics
8. Authorization

硬字段：

- Primary Research Question
- Mode
- Baseline
- Intervention
- Only Intended Scientific Difference
- Frozen Variables
- Hypothesis
- Falsification Conditions
- Primary Metric
- Mandatory Gates
- PASS / FAIL / STOP
- Codex MAY / MUST / MUST NOT
- Requires User Approval

缺关键字段：

```text
DECISION_INCOMPLETE
```

不得启动正式科研实验。

---

## 5. decision 修订规则

允许：

```text
Revision: r0 / r1 / r2 ...
```

若实验尚未产生科研结果，仅因路径、CLI、环境等非科研语义问题需要修正，可在用户与 ChatGPT 确认后修订同一 experiment。

若已经看到科研结果后需要修改：

- 模型结构；
- baseline；
- 数据；
- split；
- metric；
- threshold；
- seed plan；
- training budget；
- 主要 intervention；

则当前实验必须结束并记录真实状态，随后创建新的 experiment ID 和新的 `decision.md`。

禁止通过修改原 decision 对失败实验进行 post-hoc rescue。

---

## 6. Proposal 机制

Codex 可以主动创建：

```text
experiments/<track>/<stage>/indexN/result/proposal.md
```

建议内容：

- Observation
- Proposed Change
- Why
- Scientific Variable
- Expected Benefit
- Risks / Confounders
- Minimum Experiment
- Authorization = PENDING USER APPROVAL

Proposal 没有执行权限。

只有用户批准后，由 ChatGPT 创建新的 `decision.md`，Codex 才能执行新结构。

---

## 7. 实验模式

每轮必须声明：

```text
ENGINEERING
DIAGNOSIS
DISCOVERY
CONFIRMATION
```

### ENGINEERING

只验证：

- schema；
- CLI；
- checkpoint；
- shape；
- strict reload；
- 环境；
- 工程运行链路。

工程 PASS 不能写成科研方法有效。

### DIAGNOSIS

用于解释失败机制，例如：

- 模块是否被忽略；
- 梯度是否存在；
- 输入尺度是否异常；
- shortcut；
- deployment mismatch。

Diagnosis 不能自动授权新结构实验。

### DISCOVERY

目的：寻找值得冻结的候选结构。

默认允许：

- 单 seed；
- short run；
- Train + Dev；
- ablation；
- scale / normalization / fusion 探索；
- mechanism diagnosis。

所有结果标记：

```text
DISCOVERY_ONLY
```

不能直接当论文最终证据。

### CONFIRMATION

目的：对已冻结方案进行正式独立验证。

原则：

- 结构冻结；
- baseline 冻结；
- metric 冻结；
- PASS 条件冻结；
- split 冻结；
- checkpoint selection rule 冻结；
- 原则上多 seed；
- Final Test 仅此阶段正式使用。

若必须改结构，当前 Confirmation 结束，回到新的 Discovery。

---

## 8. Discovery → Confirmation Gate

至少满足：

- D1 Engineering PASS
- D2 Mechanism PASS
- D3 Development Behavior PASS
- D4 相对 baseline 有明确正向信号
- D5 Red-Team 无致命问题
- D6 用户明确批准

Codex 无权自行升级。

---

## 9. 核心科研原则

### 9.1 一轮一个主要科研问题

除 Integration 实验外，一轮原则上只引入一个主要 intervention。

不得未经授权同时加入：

```text
Position fusion + Force head + MASK + 新 loss + 新 augmentation
```

然后只看最终成功率。

### 9.2 公平 baseline

Candidate 与 baseline 应尽量保持：

- 同数据；
- 同 split；
- 同训练预算；
- 同 seed policy；
- 同 evaluation；
- 同 deployment；
- 同真机协议。

### 9.3 Falsification-first

每轮在运行前必须写清：

> 什么结果会证明当前假设是错的？

没有 falsification condition 的科研实验不得启动。

### 9.4 禁止 post-hoc rescue

看到结果后不得：

- 改 PASS threshold；
- 换 primary metric；
- 删除失败 episode；
- 换 Final Test；
- 临时加模块；
- 临时扩训练预算；
- 只报告最好 seed；
- 只报告最好 checkpoint。

---

## 10. 八层 Gate

### G0 — Provenance / Contract

检查：

- code state；
- dataset version；
- decision scope；
- baseline / intervention；
- reused evidence；
- workspace 状态。

### G1 — Data & Schema

检查：

- observation；
- action；
- feature order；
- cameras；
- normalization；
- split；
- train/deployment schema。

### G2 — Implementation & Regression

检查：

- tests；
- shape；
- finite output；
- strict reload；
- baseline regression；
- default-off equivalence（若适用）。

### G3 — Training Health

检查：

- loss；
- gradient；
- parameter update；
- checkpoint；
- NaN / Inf；
- processor consistency。

**G3 PASS 只说明训练工程正常。**

### G4 — Mechanism

必须直接验证：

> 新模块 / 新信号是否真的进入决策。

优先使用：

- zero；
- shuffle；
- swap；
- counterfactual；
- perturbation；
- sensitivity；
- gradient；
- signal ablation。

### G5 — Offline Behavior

检查：

> 机制产生的行为方向是否正确。

不能只看 action MAE。

### G6 — Deployment Equivalence

检查：

- processor；
- cameras；
- state order；
- inference loop；
- `n_action_steps`；
- temporal ensemble；
- clipping；
- controller ownership；

是否与真机一致。

### G7 — Real Robot

按阶段记录：

```text
Reach
Contact
Grasp
Lift
Hold
Place
Release
Spill-safe
Full Success
```

不涉及真机则：

```text
G7 = NA
```

### G8 — Scientific Conclusion

结论等级：

```text
HYPOTHESIS
OBSERVED
SUPPORTED
OFFLINE_CONFIRMED
REAL_ROBOT_CONFIRMED
REJECTED
BLOCKED
```

不得使用超过证据等级的 claim。

---

## 11. Position Gate 框架

固定层级：

```text
P0 Target Observability
P1 Mechanism
P2 Directional Behavior
P3 Local Robustness
P4 Unseen / Interpolation [optional]
```

当前项目不强制 claim 大范围空间泛化。

必须优先证明：

- 模型不是固定中央轨迹回放；
- 模型真的利用视觉目标信息；
- 局部位置变化引起正确 Reach 调整；
- 在任务定义的局部区域内具有可靠性。

具体数值阈值由对应 `decision.md` 冻结。

---

## 12. Force Gate 框架

固定层级：

```text
F0 Signal Calibration
F1 Mechanism
F2 Contact Behavior
F3 Hold Behavior
F4 Place / Release
F5 Disturbance Robustness [optional]
```

设计 Force 方法前必须先验证：

```text
gripper.pos
gripper.load
gripper.current
```

是否具有可分析的物理信息。

“加入 load/current”本身不构成 force-aware claim。

---

## 13. Integration Gate

Position 和 Force 各自阶段性成立后，最终系统必须重新验证：

```text
Local Placement
→ Reach
→ Contact
→ Grasp
→ Lift
→ Hold
→ Place
→ Release
```

Full Success 至少要求：

- 成功 Reach；
- 有效 Grasp；
- Lift；
- Hold 不掉杯；
- Place；
- Release；
- 无明显洒水。

阶段成功率必须同时报告，方便失败归因。

---

## 14. 数据与 Final Test 纪律

正式数据集必须：

```text
采集
→ QC
→ Split Freeze
→ 模型实验
```

Final Test：

- 不用于 Discovery；
- 不用于结构搜索；
- 不用于超参数选择；
- 不用于 checkpoint 选择。

任何用于模型设计的 Test 查看：

```text
TEST_CONTAMINATED
```

污染后的 Test 不得继续称为最终未见测试集。

---

## 15. Seed 原则

Discovery / Diagnosis：

```text
默认 1 seed
```

Formal Offline Confirmation：

```text
原则上 >= 3 seeds
```

除非成本有明确理由。

Real Robot：

不要求每个 seed 全部真机测试，但 seed/checkpoint selection rule 必须在正式真机结果前冻结。

---

## 16. 数据失败与任务失败

至少区分：

```text
VALID
HARDWARE_INVALID
OPERATOR_INVALID
TASK_FAILURE
```

原则：

> **TASK_FAILURE != INVALID DATA**

不得因为抓取失败、掉杯、洒水就自动删除原始证据。

---

## 17. 证据与交付

### 17.1 Codex → ChatGPT 紧凑交付

路径：

```text
experiments/<track>/<stage>/indexN/result/
```

至少：

```text
summary.md
metrics.json
validation.json
figures/        # 必要时
proposal.md     # 可选
```

### 17.2 Canonical Raw Evidence

路径：

```text
results/experiments/<experiment_id>/
```

保存：

- train logs；
- raw predictions；
- CSV；
- diagnostics；
- videos；
- checkpoint provenance；
- hashes；
- failed attempts。

Raw evidence 尽量 append-only，不覆盖失败现场。

---

## 18. Builder Review + Red-Team Review

### Builder Review

ChatGPT 第一遍检查：

> Codex 是否准确完成当前 `decision.md`？

### Red-Team Review

第二遍主动寻找：

- data leakage；
- shortcut；
- confounder；
- seed luck；
- checkpoint cherry-picking；
- post-hoc threshold；
- unfair baseline；
- deployment mismatch；
- module-not-used；
- unsupported generalization；
- test contamination；
- 不恰当删除失败数据。

---

## 19. Git / 工作区

禁止：

```text
git add .
```

Codex 必须：

- 开始和结束检查 workspace；
- 保护用户已有无关修改；
- 只 stage 本轮相关文件；
- 不覆盖用户未提交内容；
- 不删除无关未跟踪文件。

Commit：

- 可在本轮成果逻辑完整且项目规则允许时进行；
- 只包含本轮相关修改。

Push / publish：

> 必须用户明确授权。

---

## 20. 真机权限与安全

任何改变真实机械臂行为的正式实验必须由当前 `decision.md` 授权。

真机 experiment 应定义：

- workspace limit；
- max relative target；
- collision stop；
- abnormal current/load stop；
- spill stop；
- emergency stop；
- repeated failure stop；
- trial count；
- video / control log evidence。

安全停止：

```text
BLOCKED_SAFETY
```

不是科研 FAIL。

---

## 21. Codex 权限

### MAY

- 实现已批准 intervention；
- 修 bug；
- 写测试；
- 加只读诊断；
- 重构非科研语义代码；
- 跑 decision 授权实验；
- 恢复中断任务；
- 整理 evidence；
- 主动写 proposal。

### MUST

- 严格读取 decision；
- 报告 mandatory gates；
- 保留失败证据；
- 区分工程失败与科研失败；
- 记录 evidence path；
- 完成本轮后停止。

### MUST NOT

- 修改 `decision.md`；
- 自行开新科研阶段；
- 改 baseline；
- 改 split；
- 改 primary metric；
- 改 PASS threshold；
- 新增未批准结构；
- 新增未批准 loss；
- 改 observation/action semantics；
- 看 Final Test 后反向调参；
- 自动启动正式真机；
- 擅自 push/publish；
- 覆盖用户已有工作。

---

## 22. 用户批准点

必须用户批准：

- 新研究结构；
- 新主 Track；
- Discovery → Confirmation；
- 正式 Full Training；
- 正式 Final Test；
- 正式 Real-Robot Test；
- 重要采集协议变化；
- push / publish；
- 下一科研轮次。

---

## 23. 动态状态与历史记录

本文件不保存“当前正在做什么”。

动态状态写：

```text
workflow/STATUS.md
```

历史事实写：

```text
docc/日志/工作日志.md
```

单轮授权写：

```text
experiments/.../decision.md
```

稳定规则、当前状态、单轮科研合同必须分离。
