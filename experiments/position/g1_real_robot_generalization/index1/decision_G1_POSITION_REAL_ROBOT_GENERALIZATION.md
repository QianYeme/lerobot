# Experiment Decision Contract

> Revision: r0  
> 本文件由用户与 ChatGPT 冻结，Codex 只读。

# G1 — Position Real-Robot Generalization

## 1. Identity

- **Experiment ID**: `G1_POSITION_REAL_ROBOT_GENERALIZATION`
- **Track**: `POSITION`
- **Stage**: `REAL_ROBOT_GENERALIZATION`
- **Mode**: `DISCOVERY`
- **Upstream**: `G0_POSITION_GENERALIZATION_SCREENING = PASS`
- **Policy Retraining**: 禁止
- **Architecture Change**: 禁止
- **Final Test Dataset**: 禁止
- **Real Robot**: 允许
- **Primary Scientific Goal**:

> 比较 ACT baseline 与 object-aware 候选在训练工作区内不同杯位上的真实任务泛化能力，以任务成功率而不是单关节动作方向作为主要证据。

---

## 2. Frozen Shortlist

本轮只测试以下 3 个模型：

```text
B3_ACT
P1_OCACT
R3_FILM
```

角色：

```text
B3_ACT     = baseline
P1_OCACT   = best object-aware candidate by offline Dev action quality
R3_FILM    = structurally different candidate with strongest Left/Right output separation diagnostic
```

禁止临时加入：

```text
B2
P3
P4
R1
R2
```

除非另开 decision。

---

## 3. Frozen Checkpoints

每个模型必须使用 G0 中已经冻结的 checkpoint。

不得：

- 根据真机结果换 checkpoint；
- 重新训练；
- fine-tune；
- 调 hyperparameter；
- 改 object-conditioning；
- 改 prototype；
- 改 tau。

---

## 4. Deployment Contract

运行环境：

```text
target device = laptop NVIDIA RTX 4060 Laptop GPU
VRAM = 8 GB
RAM = 16 GB
```

若实际真机控制电脑不是上述设备：

> 必须记录真实 deployment hardware。

三个模型必须：

```text
same control frequency
same camera resolution
same image preprocessing
same state normalization
same action postprocessing
same safety limits
same Home pose
```

---

## 5. Scene Contract

保持：

- 同一 SO100/SO101 robot；
- 同一 Top camera；
- 同一 Wrist camera；
- 同一桌面；
- 同一透明塑料杯；
- 杯中约半杯水；
- 同一 robot base；
- camera 不移动；
- 尽量稳定照明。

禁止为某个模型单独改善：

- 光照；
- 杯子外观；
- 摄像头位置；
- Home pose；
- 工作区边界。

---

## 6. Cup Position Region

沿用正式数据采集工作区：

```text
transverse / horizontal:
40 mm total
= ±20 mm

longitudinal:
30 mm total
= ±15 mm
```

nominal center：

```text
P0
```

本轮只测试：

> 训练工作区内部的 position generalization。

本轮不测试大范围 workspace extrapolation。

---

## 7. Position Test Set

固定 12 个位置 strata，对应原数据采集的：

```text
4 × 3 grid
```

仅作为放置区域，不要求物理画网格。

横向：

```text
4 bins
```

纵向：

```text
3 bins
```

共：

```text
12 position strata
```

---

## 8. Exact Placement Policy

每次 trial：

> 在对应 stratum 内自然随机放置杯子，而不是严格放在格子中心。

要求：

- 不让模型看到 position label；
- 不使用人为提示；
- 不使用明显辅助标记作为 policy input；
- placement 误差自然存在。

---

## 9. Number of Trials

### Phase A — Safety / Function Check

每个模型：

```text
3 nominal-center trials
```

目的仅确认：

- policy 能启动；
- camera/state/action pipeline 正常；
- 无明显部署 bug；
- 无危险动作。

这些 trial：

```text
NOT counted in primary success rate
```

如果某模型连 nominal center 都因工程问题无法运行：

```text
BLOCKED_DEPLOYMENT
```

先修工程问题，不作为科学 FAIL。

---

### Phase B — Generalization Test

每个模型：

```text
12 strata × 2 trials = 24 trials
```

总计：

```text
3 models × 24 = 72 formal trials
```

这是建议主设计。

若时间允许，可升级：

```text
12 strata × 3 trials = 36/model
```

但必须三个模型全部采用相同 trial 数。

禁止某个模型失败后少测，另一个成功后多测。

---

## 10. Randomization

不得按模型连续跑完整 24 次。

推荐按 block：

```text
Block 1:
B3 → P1 → R3

Block 2:
R3 → B3 → P1

Block 3:
P1 → R3 → B3
```

模型顺序轮换。

每个模型内部：

```text
12 strata random order
```

第二轮重复时重新随机。

目的：

> 降低时间、照明、机械臂温度、操作者放杯习惯等 confound。

---

## 11. Reset Policy

每个 formal trial 前：

```text
robot → same Home + Open
cup → selected random position
water level → approximately half
scene → visually checked
```

模型不得继承上一 trial 的隐藏状态。

如果 policy 有 temporal buffer / action queue：

```text
clear/reset before each trial
```

---

## 12. Autonomous Rollout

每次 trial：

```text
Home + Open
→ autonomous Reach
→ autonomous Fine Alignment
→ autonomous Grasp
→ autonomous Lift
→ autonomous Hold
→ autonomous Lower
→ autonomous Release
→ autonomous Retreat / Home
```

人类不得中途修正。

唯一允许人工介入：

```text
SAFETY_ABORT
```

---

## 13. Success Labels

每个 trial 必须记录以下分阶段结果。

### 13.1 Reach Success

满足：

> gripper 到达可以合理抓取杯子的局部位置，没有明显去固定中心/错误目标。

记录：

```text
REACH = 0/1
```

### 13.2 Grasp Success

满足：

> gripper 正确闭合并形成有效夹持。

```text
GRASP = 0/1
```

### 13.3 Lift Success

满足：

> 杯子脱离桌面并保持稳定至少约 0.5 s。

```text
LIFT = 0/1
```

这是 Position Track 最重要的中间指标之一。

### 13.4 Hold Success

满足：

> lift 后短暂停留时杯子未立即滑落/弹出。

```text
HOLD = 0/1
```

### 13.5 Place Success

满足：

> 杯子被重新放回桌面且未倾倒。

```text
PLACE = 0/1
```

### 13.6 Overall Success

严格定义：

```text
OVERALL = 1
```

仅当：

```text
Reach
+ Grasp
+ Lift
+ Hold
+ Lower
+ Release/Place
```

完整成功。

---

## 14. Position-Specific Primary Metrics

每个模型必须报告：

```text
Overall Success Rate
Reach Success Rate
Grasp Success Rate
Lift Success Rate
Place Success Rate
```

同时按：

```text
12 strata
Left half vs Right half
Near row vs Middle row vs Far row
Edge strata vs Interior strata
```

分别统计。

---

## 15. Primary Scientific Metric

本轮 Position Track 的第一主指标：

```text
LIFT SUCCESS RATE
```

原因：

> 能稳定把杯子从不同位置抓起，直接反映目标定位 + reach + grasp 是否形成有效任务行为。

第二主指标：

```text
OVERALL TASK SUCCESS RATE
```

---

## 16. Why Lift Is Primary

完整 place/release 还会受到：

- Force/control；
- release timing；
- placement strategy；

影响。

因此 Position 模块不能因为后半段 Force 失败而被完全否定。

若：

```text
Reach + Grasp + Lift success
```

但之后 release/drop 失败：

> Position 仍可视为成功完成其核心职责。

---

## 17. Failure Taxonomy

每个失败 trial 必须分类：

```text
F1_TARGET_MISS
F2_REACH_CENTER_BIAS
F3_BAD_ALIGNMENT
F4_GRASP_MISS
F5_GRASP_BUT_DROP_DURING_LIFT
F6_HOLD_DROP
F7_PLACE_FAILURE
F8_RELEASE_FAILURE
F9_SAFETY_ABORT
F10_DEPLOYMENT_ERROR
```

允许：

```text
primary failure + secondary notes
```

---

## 18. Center-Bias Real-Robot Diagnostic

重点观察：

> 杯子在不同位置时，模型是否仍趋向 nominal center。

若出现：

```text
cup shifted
but trajectory repeatedly converges toward P0
```

记录：

```text
CENTER_BIAS_EVENT = TRUE
```

统计每模型：

```text
center-bias event rate
```

这是比离线 action-direction 更直接的泛化证据。

---

## 19. Object-Aware Candidate Diagnostics

对于：

```text
P1_OCACT
R3_FILM
```

正式 primary trials 使用：

```text
Normal mode only
```

本轮不在同一批 formal trials 中做：

```text
zero object
shuffle object
```

避免扩大真机实验量。

如果 P1/R3 明显优于 ACT：

> 后续可单独创建 mechanism real-robot ablation decision。

---

## 20. Safety Abort

出现以下情况立即人工停止：

- 机械臂可能撞击桌面/相机/底座；
- action 持续发散；
- gripper 挤压杯子造成明显破坏风险；
- 杯子即将倾倒到电子设备方向；
- robot state/action communication 异常。

Safety abort：

```text
不等同普通 task failure
```

必须单独记录。

---

## 21. Deployment Error Rule

若失败由：

- checkpoint load；
- camera key mismatch；
- state schema；
- latency pipeline；
- normalization config；
- code exception；

造成：

```text
DEPLOYMENT_ERROR
```

该 trial 不计入科学成功率。

修复后必须重新执行对应 trial。

但禁止借工程修复改变模型科研结构。

---

## 22. Runtime Logging

每个模型记录：

```text
mean inference latency
P95 latency
peak VRAM
RAM usage if available
control-loop frequency
dropped frames
```

用于后续 G6 Deployment Equivalence。

---

## 23. Video Evidence

每个 formal trial 推荐保存：

```text
Top video
Wrist video
optional external overview video
```

至少需要：

```text
trial_id
model
position_stratum
trial_repeat
result labels
```

与视频一一对应。

---

## 24. Blinding / Labeling

为了降低事后主观解释：

trial 执行后立即填写：

```text
Reach
Grasp
Lift
Hold
Place
Overall
Failure Type
```

不要等看完所有模型结果后再统一回忆标注。

---

## 25. Comparison

主要比较：

```text
B3_ACT vs P1_OCACT
B3_ACT vs R3_FILM
P1_OCACT vs R3_FILM
```

报告：

```text
absolute success difference
relative success difference
```

样本数较小时优先报告：

```text
raw counts + percentages
```

并可附：

```text
Wilson confidence interval
```

不强制复杂显著性检验。

---

## 26. PASS Logic

本轮不是要求所有 object-aware 模型都 PASS。

### Candidate Position PASS

某候选满足以下模式即可判为有价值：

1. Lift success 在 12 个位置 strata 中稳定；
2. edge/shifted positions 不出现明显 collapse；
3. center-bias event rate 低；
4. 至少不明显劣于 ACT baseline；
5. 若明显优于 ACT，则形成强 Position evidence。

### Strong PASS

若 object-aware candidate：

```text
Lift / Overall success
```

明显高于 ACT，且优势主要来自：

```text
shifted / edge positions
```

则：

> 强支持 object-aware position generalization hypothesis。

---

## 27. Candidate Selection After G1

### Case A — P1 Best

```text
freeze P1
```

进入：

```text
Position Confirmation / Force Integration
```

### Case B — R3 Best

```text
freeze R3
```

进入后续阶段。

### Case C — ACT Best / Tie

若 object-aware 模型没有真机优势：

> 不因为离线机制漂亮而强行保留。

此时优先：

- ACT baseline；
- 或重新评估是否 Position module 值得继续。

### Case D — All Models Poor

如果三个模型在多数位置都抓不起来：

> 先检查训练数据/task supervision/deployment，而不是继续 architecture search。

---

## 28. Final Test Dataset

本轮仍然：

```text
Final Test = untouched
```

原因：

> G1 是真实机器人 Discovery/generalization screening，不是最终离线 Confirmation。

最终 paper evidence 是否需要使用 Final Test，由后续 Confirmation decision 冻结。

---

## 29. Required Outputs

```text
experiments/position/g1_real_robot_generalization/index1/
├── decision.md
└── result/
    ├── summary.md
    ├── validation.json
    ├── trial_manifest.json
    ├── trial_results.csv
    ├── trial_results.json
    ├── failure_analysis.json
    ├── position_breakdown.json
    ├── resource_metrics.json
    ├── selected_candidate.json
    ├── videos/
    │   ├── B3_ACT/
    │   ├── P1_OCACT/
    │   └── R3_FILM/
    ├── figures/
    │   ├── lift_success_by_model.*
    │   ├── overall_success_by_model.*
    │   ├── success_by_position.*
    │   ├── edge_vs_interior.*
    │   ├── failure_types.*
    │   └── center_bias_events.*
    └── proposal.md
```

---

## 30. Required Summary Table

最终必须生成：

```text
Model
Trials
Reach Success
Grasp Success
Lift Success
Hold Success
Place Success
Overall Success
Edge Lift Success
Center-Bias Events
Safety Aborts
Mean Latency
Peak VRAM
```

---

## 31. Codex MAY

- 实现真机 trial runner；
- 实现 trial reset；
- 记录视频/日志；
- 随机化 position order；
- 自动保存 trial metadata；
- 统计成功率；
- 生成图表；
- 创建 proposal.md。

---

## 32. Codex MUST

- 只测试 B3/P1/R3；
- 使用冻结 checkpoint；
- 每模型同等 trial 数；
- position randomization；
- trial 前 reset；
- 明确 phase success；
- Lift 作为 Position 主指标；
- Final Test 不读；
- 完成后停止。

---

## 33. Codex MUST NOT

- 修改本 decision.md；
- 重训模型；
- 根据真机结果换 checkpoint；
- 改 prototype/tau；
- 改 architecture；
- 调模型超参数；
- 人工修正 autonomous rollout；
- 使用 Final Test；
- 自动进入 Force integration；
- 自动进入 multi-seed Confirmation。

---

## 34. Completion Rule

完成后必须输出：

```text
POSITION_REAL_ROBOT_GENERALIZATION_COMPLETE = TRUE
```

并给出：

```text
selected_candidate
```

但不得自动进入下一阶段。

由用户 + ChatGPT 根据真实：

```text
Reach / Grasp / Lift / Overall success
```

决定最终 Position 模型及后续 Force Track。
