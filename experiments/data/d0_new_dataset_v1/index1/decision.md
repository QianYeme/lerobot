# Experiment Decision Contract

> Revision: r0
> 本文件由用户与 ChatGPT 冻结，Codex 只读。

---

## 1. Identity & Mode

- **Experiment ID**: `D0_NEW_DATASET_V1`
- **Track**: `DATA`
- **Stage**: `NEW_DATASET_COLLECTION`
- **Mode**: `ENGINEERING`
- **Target Dataset Size**: `60 episodes`
- **Primary Object**: 半满透明塑料杯
- **Model Training**: 禁止
- **Formal Model Evaluation**: 禁止
- **Real-Robot Operation**: 仅限数据采集，不属于正式模型真机评估

### Current Authorization

本轮只授权：

> 建立新的 60-episode 半满透明塑料杯抓取数据集，确保局部位置变化、动作阶段语义和 gripper force-related signals 足够清晰，供后续 ACT Baseline、Position、Force 与 Integration 实验使用。

本轮明确不授权：

- 训练 ACT；
- 训练 Position 模块；
- 训练 Force 模块；
- 运行旧 C1；
- 正式 Final Test；
- 正式模型真机比较；
- 修改本 decision 中已冻结的采集语义。

---

## 2. Research Question

> **RQ:** 能否建立一个具有真实局部位置变化、清晰抓取阶段语义、稳定 Hold supervision，并完整保留 gripper `pos/load/current` 的正式新数据集，使后续 Position 与 Force 研究能够进行受控比较？

本轮不是模型效果实验。

---

## 3. Paper Relevance

### Why This Dataset Is Needed

旧实验暴露了两类问题：

1. Position：策略倾向名义中心轨迹，目标位置变化利用不足；
2. Force / Hold：旧 demonstration 的 lift/place 段存在 gripper 逐渐回开，造成 imitation supervision 含糊。

F0-LEGACY 已观察到：

- `gripper.load` 在 closing / release 转换上具有稳定信息；
- 单帧 load 不足以可靠表示 stable hold；
- 时间窗口更有价值；
- 旧 demonstrations 在 lift/place 中存在明显 reopen 行为。

因此新数据必须同时改善：

- local position coverage；
- phase separation；
- Hold 阶段 gripper supervision。

### Non-Claims

本轮 PASS 不能证明：

- Position 方法有效；
- Force-aware policy 有效；
- DINO / object-centric 方法有效；
- 真机完整任务已解决；
- 大范围空间泛化成立。

---

## 4. Task Contract

完整 episode：

```text
Home + Open
→ Reach
→ Fine Alignment
→ Pre-Grasp Stabilization
→ Grasp
→ Grasp Stabilization
→ Lift
→ Hold
→ Lower
→ Place Stabilization
→ Release
→ Release Stabilization
→ Retreat
→ Home + Open
```

### Start State

- 固定 Home pose；
- gripper = OPEN；
- 稳定约 0.3–0.5 s。

### End State

- Retreat；
- 返回 Home；
- gripper = OPEN；
- 稳定约 0.3–0.5 s。

---

## 5. Scene Contract

### Object

- 同型号透明塑料杯；
- 近似半杯水；
- 杯体竖直；
- 不新增高对比视觉标记。

### Nominal Grasp Region

第一版目标区域：

- 左右约 ±20 mm；
- 前后约 ±15 mm；
- 总体约 40 × 30 mm。

该尺寸属于 nominal design。

最终论文中使用的有效范围应由数据采集后的实测位置分布确定，而不是为了满足合同强行凑满 40 × 30 mm。

### Region Marking

允许：

- 四角小型低反光标记；
- 相机画面外或低显著度辅助边界。

避免：

- 高对比完整矩形；
- 明显反光透明胶带形成视觉 shortcut；
- 在桌面画出 4×3 内部网格供模型看到。

---

## 6. Spatial Sampling Contract

将 nominal region 在采集管理上划分为：

```text
4 columns × 3 rows = 12 spatial strata
```

这些 strata 不是 12 个固定抓取点。

每个 stratum：

```text
5 episodes
```

总计：

```text
12 × 5 = 60 episodes
```

### Within-Stratum Placement

每次在对应 stratum 内人工自然摆放。

禁止：

- 每次精确放在格子中心；
- 用固定机械模板复制相同坐标；
- 让一个 stratum 的 5 条轨迹几乎完全相同。

### Collection Rounds

推荐：

```text
Round 1: 每个 stratum 1 条
Round 2: 每个 stratum 1 条
Round 3: 每个 stratum 1 条
Round 4: 每个 stratum 1 条
Round 5: 每个 stratum 1 条
```

每一 round 内随机打乱 12 个 stratum 的采集顺序。

禁止：

```text
A1 连采 5 条
→ A2 连采 5 条
→ ...
```

---

## 7. Motion Phase Contract

### S0 — Home/Open

- Home pose；
- gripper OPEN；
- 0.3–0.5 s 稳定。

### S1 — Reach

- 从 Home 向杯子运动；
- gripper 全程保持 OPEN；
- 杯子偏左/右/前/后时，整条 reach trajectory 应自然产生对应调整；
- 禁止所有轨迹先走固定中心，再最后突然修正。

### S2 — Fine Alignment

- 最后 30–50 mm 左右精确靠近；
- gripper 仍保持 OPEN；
- 不与 closing 重叠。

### S3 — Pre-Grasp Stabilization

- 到达抓取位置；
- 机械臂基本静止；
- 约 0.2–0.3 s。

### S4 — Grasp

- 机械臂尽量保持静止；
- 主要进行 gripper closing；
- 建立有效杯壁接触。

### S5 — Grasp Stabilization

- 闭爪完成后保持；
- 约 0.5–0.8 s；
- 推荐约 0.6 s；
- 此阶段禁止开始 Lift；
- 禁止重新张爪。

### S6 — Lift

- 近似垂直抬升；
- 第一版目标高度约 60 mm；
- 推荐约 0.6–1.0 s；
- **gripper target 必须基本保持稳定闭合**。

### S7 — Hold

- 悬停约 1.0 s；
- **gripper target 保持稳定闭合**；
- 禁止人为逐渐张开；
- 不主动调整成 release 动作。

### S8 — Lower

- 稳定下降；
- gripper 继续保持稳定抓持；
- 禁止边下降边松开。

### S9 — Place Stabilization

- 杯底重新接触桌面后；
- gripper 仍保持抓持；
- 约 0.3–0.5 s；
- 推荐约 0.4 s。

### S10 — Release

- 机械臂基本稳定；
- gripper 打开。

### S11 — Release Stabilization

- 原地保持 OPEN；
- 约 0.3–0.5 s；
- 推荐约 0.4 s。

### S12 — Retreat

- 沿安全方向退出约 50–80 mm；
- gripper 保持 OPEN。

### S13 — Return Home

- 返回统一 Home；
- gripper 保持 OPEN；
- episode 结束前稳定约 0.3–0.5 s。

---

## 8. Gripper Supervision Contract

这是本数据集相对旧数据的关键修正之一。

### Mandatory Rule

从 `Grasp Stabilization` 开始，到 `Place Stabilization` 结束：

```text
gripper target 应基本保持稳定
```

不得出现旧数据中常见的：

```text
Grasp
→ Lift
→ Hold
→ gripper target 逐渐回开
```

除非发生安全原因或明确 operator invalid。

### Why

F0-LEGACY 已观察到：

- 旧 demonstration 的 lift/place 阶段存在明显 reopen；
- 约 25% episode 的 reopen 幅度达到约 5° 或以上；
- 这可能直接给 imitation policy 提供错误 Hold supervision。

因此新数据必须明确：

> Hold 阶段的 expert intention 是“持续抓持”，不是“逐渐释放”。

---

## 9. Force-Related Signal Contract

每帧必须继续记录：

- `gripper.pos`
- `gripper.load`
- `gripper.current`

若现有 LeRobot schema 支持，同时保留：

- gripper action / target
- arm state
- timestamps / frame index

### Important Interpretation

本轮不把 `load` 直接定义为真实物理夹持力。

当前只视为：

```text
force-related / load proxy signal
```

`current` 当前也不得假定为可靠主信号。

本轮任务是保证新数据中这些信号完整、同步、可用于后续 F0-v2 分析。

---

## 10. Allowed Natural Variation

允许并希望保留：

- 杯子在局部 region 内的小范围位置差异；
- 遥操作的自然小幅轨迹差异；
- 正常反光变化；
- 正常 contact / load 差异。

不应人为消除成完全重复轨迹。

---

## 11. Frozen Variables

正式采集开始后默认冻结：

### Hardware / Scene
- robot base
- cameras
- camera placement
- table
- cup type
- approximate water level
- nominal Home pose

### Data
- FPS
- image resolution
- observation schema
- action schema
- load/current acquisition path

### Task
- phase order
- gripper phase semantics
- nominal lift height
- local placement region

未经用户 + ChatGPT 修订 decision，不得中途改变。

---

## 12. Episode Status

每条 episode 必须分类：

```text
VALID
HARDWARE_INVALID
OPERATOR_INVALID
TASK_FAILURE
```

### HARDWARE_INVALID

例如：

- 相机丢帧；
- 通信中断；
- 数据损坏；
- episode 截断。

### OPERATOR_INVALID

例如：

- 忘记 Hold；
- 提前 Release；
- 杯子明显摆错区域；
- Lift/Hold 阶段明显主动回开 gripper；
- 明显违反 phase protocol。

### TASK_FAILURE

例如：

- 没抓准；
- 滑杯；
- 掉杯；
- 洒水。

原则：

```text
TASK_FAILURE != INVALID DATA
```

失败 episode 不得因为“不漂亮”自动删除。

---

## 13. Data Quality Gates

### D-G0 — Schema / Integrity

检查：

- episode 可读；
- cameras 完整；
- state/action 完整；
- `pos/load/current` 完整；
- timestamp/frame 顺序正确；
- 无大规模 NaN / Inf。

### D-G1 — Spatial Coverage

采集后使用可信的杯中心估计或离线分析得到：

```text
(cx, cy)
```

并检查：

- 12 个 strata 都有覆盖；
- 没有严重中心聚集；
- 没有明显分成旧 Left/Center/Right 三簇；
- 实测局部变化真实存在。

### D-G2 — Motion Phase Quality

检查：

- Reach 与 Grasp 可区分；
- Grasp 与 Lift 不严重重叠；
- Hold 明确存在；
- Lower 与 Release 不严重重叠；
- episode start/end 语义一致。

### D-G3 — Hold Supervision Quality

重点检查：

- Grasp Stabilization → Lift → Hold → Lower → Place Stabilization 期间 gripper target 是否稳定；
- 是否存在系统性 reopen；
- reopen episode 数量和幅度。

若仍大量存在旧模式，本轮数据集不得直接冻结。

### D-G4 — Force Signal Integrity

检查新数据中的：

- `pos`
- `load`
- `current`
- gripper action

是否完整并具有时间结构。

本 Gate 不要求现在完成 Force classifier。

### D-G5 — Task Execution Summary

逐 episode 记录：

- Reach
- Grasp
- Lift
- Hold
- Place
- Release
- Drop
- Spill

---

## 14. Required QC Outputs

完成采集后，Codex 应生成：

```text
dataset_review/
├── validation.json
├── spatial_report.json
├── motion_phase_report.json
├── gripper_hold_report.json
├── force_signal_report.json
└── figures/
    ├── cup_position_scatter.*
    ├── phase_duration_distribution.*
    ├── gripper_action_hold_distribution.*
    ├── example_force_timeline.*
    └── invalid_episode_summary.*
```

---

## 15. Split Rule

本轮采集和 QC 完成前：

```text
禁止生成最终 Train / Dev / Final Test
```

只有 Dataset Review PASS 后，才生成：

```text
split_manifest_v1.json
```

目标：

```text
Train 36
Dev   12
Final Test 12
```

每个 spatial stratum：

```text
3 Train
1 Dev
1 Test
```

同时检查采集 Round 的时间分布，避免 Test 全来自后期熟练阶段。

Split 必须在任何正式模型结果产生前冻结。

---

## 16. PASS / FAIL / BLOCKED

### PASS

满足：

- 目标 60 条数据基本完成；
- schema 与 signals 完整；
- spatial coverage 符合局部区域设计；
- motion phases 清晰；
- Hold supervision 明显优于旧数据；
- Lift/Hold 不存在系统性 gripper reopen；
- force-related signals 可用于后续分析；
- QC 完成。

### FAIL

例如：

- 实际杯位几乎没有变化；
- 轨迹仍明显向固定中心收缩；
- phase 大量重叠；
- Hold 阶段仍系统性回开；
- force signals 大量缺失；
- 场景设计产生明显视觉 shortcut。

### BLOCKED

例如：

- 硬件；
- camera；
- telemetry；
- 数据采集工具；
- 安全问题；

导致无法完成有效采集。

---

## 17. Codex MAY

- 检查并完善数据采集辅助工具；
- 编写 episode QC；
- 编写 spatial coverage 分析；
- 编写 motion phase 分析；
- 编写 gripper Hold 稳定性分析；
- 编写 force signal 完整性分析；
- 生成图表；
- 整理 evidence；
- 创建 `proposal.md`。

---

## 18. Codex MUST

- 不覆盖 raw dataset；
- 保留 invalid / failure 记录；
- 区分 TASK_FAILURE 和 INVALID；
- 明确自动推断与人工确认的区别；
- 保存 QC evidence；
- 报告违反 Hold supervision 的 episode；
- 完成本轮后停止。

---

## 19. Codex MUST NOT

- 修改本 `decision.md`；
- 训练任何正式模型；
- 删除失败 episode 以提高数据质量统计；
- 修改 spatial split 迎合后续模型；
- 擅自改变 phase timing / semantics；
- 擅自改变 nominal region；
- 自动进入 ACT Baseline；
- 自动进入 Position；
- 自动进入 Force；
- 自动启动正式真机比较。

---

## 20. Required Deliverables

本轮目录：

```text
experiments/data/d0_new_dataset_v1/index1/
├── decision.md
└── result/
    ├── summary.md
    ├── metrics.json
    ├── validation.json
    ├── figures/
    └── proposal.md       # optional
```

Raw / QC evidence：

```text
results/experiments/D0_NEW_DATASET_V1/
```

---

## 21. Completion Rule

完成采集与 QC 后停止。

只有用户 + ChatGPT 验收数据集并明确批准后，才允许：

```text
Dataset Freeze
→ Split Freeze
→ New ACT Baseline
```

不得自动进入模型训练。
