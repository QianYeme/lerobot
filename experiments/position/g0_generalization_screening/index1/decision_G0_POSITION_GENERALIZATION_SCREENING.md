# Experiment Decision Contract

> Revision: r0  
> 本文件由用户与 ChatGPT 冻结，Codex 只读。  
> 本 decision 正式取代 H1 作为当前 Position Track 的下一阶段入口。  
> 原 H0/H1 的 horizontal-action diagnostic 保留为辅助分析，不再作为模型淘汰硬门槛。

# G0 — Position Generalization Screening

## 1. Identity

- **Experiment ID**: `G0_POSITION_GENERALIZATION_SCREENING`
- **Track**: `POSITION`
- **Stage**: `GENERALIZATION_SCREENING`
- **Mode**: `DISCOVERY`
- **Policy Retraining**: 禁止
- **Architecture Change**: 禁止
- **Final Test**: 暂时禁止
- **Real Robot**: 本轮主筛选阶段禁止；只有完成候选排名后才进入下一份真机 decision

---

# 2. Revised Scientific Goal

Position Track 的核心问题正式定义为：

> **模型在训练区域内不同杯子位置上，是否能够根据视觉目标位置调整行为，并最终正确完成抓取任务。**

不再要求：

> 杯子横移必须对应某一个固定 action dim、固定 h0 动作方向或高 horizontal-direction consistency。

原因：

- 当前杯位横向跨度仅约 ±20 mm；
- 完整抓取轨迹由 Reach / Fine Alignment / Grasp / Lift / Lower / Release 等多个阶段组成；
- 小范围位置变化可能通过多关节、不同时间点、不同轨迹方式完成；
- “单关节横向方向一致性”不是任务成功的必要条件。

因此：

```text
Task-level generalization > single-action interpretability
```

---

# 3. Frozen Dataset

正式数据：

```text
formal4/kind_merged_nomaster
```

冻结 split：

```text
Train = 36
Dev = 12
Final Test = 12
```

本轮只允许：

```text
Train
Dev
```

Final Test 继续封存。

---

# 4. Frozen Candidate Models

本轮只评估已有模型，不重新训练。

至少包括当前可用的：

```text
B0 / B3 equivalent ACT baseline
P1 original OC-ACT
T2 / P3 fixed-scale spatial
T3 / P4 joint state-spatial
N0 R1 relational
N0 R2 action-level spatial
N0 R3 FiLM
```

若某 checkpoint 缺失或不可加载：

```text
mark unavailable
```

不得重训补齐。

---

# 5. Candidate Selection Source

每个模型使用其各自 Discovery 阶段已经冻结的：

```text
best Dev checkpoint
```

禁止：

- 根据本轮位置分组结果重新挑 checkpoint；
- 为边缘位置另选 checkpoint；
- 为某模型单独换训练配置。

---

# 6. Position Grouping

使用现有冻结 cup position estimator：

```text
P0.1 global soft center
x_global
y_global
```

仅用于分组和分析。

---

## 6.1 Horizontal Groups

按 Dev 中 cup x 分布预先划分：

```text
Left
Center
Right
```

推荐：

```text
tertiles
```

即按 Dev 样本位置分位数分 3 组。

不得看到模型结果后调整边界。

---

## 6.2 Longitudinal Groups

同样按 y 分成：

```text
Near
Center
Far
```

使用固定 tertiles。

---

## 6.3 Edge Group

定义位置分布中：

```text
|x - median_x|
or
|y - median_y|
```

最大的样本为 Edge group。

推荐使用：

```text
top 25%
```

作为边缘样本。

该定义在模型评估前冻结。

---

# 7. Primary Offline Metrics

每个模型统一报告：

```text
Overall Dev inference L1
Arm-only L1
Gripper-only L1
h0 L1
Per-action-dimension L1
```

但这些不再是唯一主指标。

---

# 8. Position-Stratified Metrics

对以下分组分别报告：

```text
Left / Center / Right
Near / Center / Far
Edge / Non-edge
```

每组至少统计：

```text
arm L1
gripper L1
h0 arm L1
full-chunk arm L1
```

重点关注：

> 模型是否在边缘位置明显退化，或是否对不同位置表现趋同。

---

# 9. Center-Bias Diagnostic

必须检查模型是否出现：

> 无论杯子在哪里，都倾向输出接近“中心杯位”的平均动作。

---

## 9.1 Predicted Action Diversity

对 Dev Reach / Fine Alignment 阶段：

比较不同位置组：

```text
predicted action distribution
```

报告：

```text
between-group distance
within-group variance
```

若不同杯位组的 predicted action 几乎完全重叠：

```text
CENTER_BIAS_RISK = TRUE
```

---

## 9.2 Position-Conditioned Prediction Difference

比较：

```text
Left vs Right
Near vs Far
```

的 predicted action chunk。

不要求某单一 joint 符号固定，只要求：

> 模型输出能够随 cup position 发生稳定、可重复的整体差异。

可使用：

```text
L2 distance
cosine distance
multi-dimensional action difference
```

不得强制使用 dim0。

---

# 10. Matched-State Generalization Check

为了减少 robot state confounding：

在 Dev 中构造尽可能多的：

```text
similar robot state
similar task phase
different cup position
```

frame pairs。

对每个模型比较：

```text
predicted action difference
```

目标不是判断“方向对不对”，而是判断：

> 不同目标位置是否引发不同策略输出。

如 Dev 可配对样本很少：

```text
report limitation
```

不得因此判 FAIL。

---

# 11. Full-Chunk Analysis

ACT 输出 action chunk 时，本轮必须分析：

```text
full predicted chunk
```

而不是只看 h0。

至少报告：

```text
first 1
first 5
first 10
full available chunk
```

的：

```text
position-conditioned action difference
```

原因：

> 小范围杯位变化的修正可能不发生在第一帧 action。

---

# 12. Phase-Specific Analysis

优先分析：

```text
Reach
Fine Alignment
Pre-Grasp
Grasp
Lift
```

其中 Position Track 的主关注阶段：

```text
Reach
Fine Alignment
Pre-Grasp
```

但最终候选选择不能只看这些阶段。

还要检查：

```text
Grasp
Lift
```

是否明显退化。

---

# 13. Object-Conditioning Mechanism

机制分析保留为辅助证据。

对 object-aware 模型继续运行：

```text
Normal
Zero object
Shuffle object
```

若模型在：

```text
Zero / Shuffle
```

时明显退化：

> 支持模型实际利用 object information。

但：

> object ablation 不再作为 Position candidate 的硬 PASS 门槛。

---

# 14. Explicit Spatial Ablations

对于有 explicit spatial branch 的模型：

```text
T2
T3
R1
R2
R3
```

允许复用已有：

```text
spatial zero
spatial shuffle
x/y transplant
```

结果。

不要求重新运行，除非现有结果格式无法比较。

这些结果仅作为：

```text
mechanism support
```

不再直接决定 candidate 淘汰。

---

# 15. Candidate Ranking

模型排序固定按以下优先级：

## Rank 1 — Position-Generalization Robustness

重点：

```text
Edge-group performance
Left/Right balance
Near/Far balance
```

避免模型只在中心位置表现好。

## Rank 2 — Overall Dev Action Quality

```text
overall arm/action error
```

## Rank 3 — Grasp/Lift Preservation

不能为了位置变化而破坏：

```text
gripper control
lift behavior
```

## Rank 4 — Center-Bias Risk

优先选择：

```text
position-conditioned outputs clearly differ
```

且没有明显 regression-to-center 的模型。

## Rank 5 — Mechanism Evidence

object/spatial ablation 作为加分项。

## Rank 6 — Resource Cost

若性能接近，优先：

```text
smaller / faster / simpler
```

---

# 16. No Artificial Horizontal Gate

明确禁止继续用以下规则淘汰模型：

```text
horizontal directional consistency < 0.70
→ FAIL
```

旧指标只能保留在附录诊断中。

---

# 17. Shortlist Rule

本轮最终选择：

```text
2–3 candidates
```

进入真机 Position Generalization Test。

至少包括：

```text
1 × ACT baseline
1–2 × best object-aware candidates
```

即使 baseline offline 不是最好，也必须保留一个 baseline 做真机对照。

---

# 18. Suggested Shortlist Size

推荐：

```text
3 models total
```

例如：

```text
ACT baseline
best OC-ACT family candidate
second-best structurally different candidate
```

Codex 不得自动开始真机测试。

---

# 19. Real-Robot Stage Preview

下一阶段将单独创建：

```text
G1_POSITION_REAL_ROBOT_GENERALIZATION
```

核心评价：

```text
random cup positions
→ autonomous rollout
→ reach success
→ grasp success
→ lift success
→ place success
→ overall success
```

真机成功率将成为 Position Track 的核心结论。

---

# 20. Required Outputs

```text
experiments/position/g0_generalization_screening/index1/
├── decision.md
└── result/
    ├── summary.md
    ├── validation.json
    ├── candidate_table.json
    ├── position_groups.json
    ├── stratified_metrics.json
    ├── center_bias.json
    ├── matched_state_analysis.json
    ├── chunk_analysis.json
    ├── mechanism_summary.json
    ├── shortlist.json
    ├── figures/
    │   ├── dev_overall_comparison.*
    │   ├── left_center_right.*
    │   ├── near_center_far.*
    │   ├── edge_vs_nonedge.*
    │   ├── action_chunk_position_difference.*
    │   ├── center_bias.*
    │   └── candidate_ranking.*
    └── proposal.md
```

---

# 21. Required Candidate Summary Table

必须生成：

```text
Model
Overall Dev L1
Arm L1
Gripper L1
Left L1
Center L1
Right L1
Near L1
Far L1
Edge L1
Center Bias Risk
Object/Spatial Usage
Params
Latency if available
Rank
```

---

# 22. Codex MAY

- 读取已有 checkpoints；
- 使用 Train/Dev；
- 执行统一离线 inference；
- 分组分析；
- 分析 full action chunk；
- 做 matched-state pair；
- 复用已有 ablation；
- 生成 shortlist；
- 生成图表；
- 创建 proposal.md。

---

# 23. Codex MUST

- 不重训模型；
- 不改架构；
- 不读 Final Test；
- 不再使用单一 dim0/h0 作为硬 gate；
- 保留 ACT baseline；
- 选择 2–3 个候选；
- 按 task-level generalization 排名；
- 完成后停止。

---

# 24. Codex MUST NOT

- 修改本 decision.md；
- 新增 architecture；
- 重新训练已有模型；
- 使用 Final Test；
- 启动 real robot；
- 因横向 action direction 不一致淘汰模型；
- 自动进入 Confirmation；
- 自动开始 Force Track。

---

# 25. Completion Rule

最终输出：

```text
POSITION_GENERALIZATION_SHORTLIST_READY = TRUE
```

并给出：

```text
2–3 model shortlist
```

然后停止。

下一步由用户 + ChatGPT 创建：

```text
G1 — Position Real-Robot Generalization
```

以真实抓取成功率作为 Position Track 的核心验证。
