# F0 Legacy Force Calibration — Result

## Outcome

**Overall: PARTIAL.**

旧 Formal3 数据中的 `gripper.load` 含有可重复的**运动/接触转换瞬态**，但没有证据表明它的单帧绝对值能表示稳定夹持力；`gripper.current` 分辨率低且阶段区分弱。`load` 的短时间窗口特征比单帧值明显更有价值，因此下一轮若获授权，应只把它当作带方向与历史的接触状态信号，而不是物理力标定值。

本轮没有训练模型、修改数据、修改 ACT、运行控制器或操作真机。

## Executed scope

- 数据真源：`数据集/formal3/kind_merged`，60 episodes、35,917 frames、30 FPS。
- 使用原始 9D state，保留并分别读取 `gripper.pos/load/curr/master_gripper.pos`；gripper action 来自 6D action 的 `gripper.pos`。
- 人工证据：复用 60/60 已审核的 `closing_start/closing_end/lift_start/release_start/release_end`。
- 自动派生粗阶段：`free_open`、`closing_contact`、`stable_grasp`、`lift_place`、`release`。
- `closing_contact` 不能可靠拆成 S1/S2；`lift_place` 不能可靠拆成 S4/S5，均记录为 `UNRESOLVED_PHASE`。
- 单帧/阶段值与 15 帧（0.5 s）窗口的 mean/std/slope/delta/min/max 均已统计。

复现实验：

```powershell
python scripts/analyze_legacy_force_calibration.py `
  --dataset 数据集/formal3/kind_merged `
  --annotations outputs/formal3_phase_labels_20260927/final_reviewed_labels/reviewed_phase_annotations.json `
  --result experiments/force/f0_legacy_calibration/index1/result `
  --raw results/experiments/F0_LEGACY_FORCE_CALIBRATION
```

## Key evidence

### Signal availability and quality

四个必需信号均为 35,917/35,917 有限值，无缺失、NaN 或 Inf。

- `load`：202 个离散值，范围 `[-448, 500]`，全局极值帧占 0.086%；未见大面积饱和。
- `current`：仅 18 个离散值，范围 `[0, 19]`，处于全局最小/最大值的帧占 30.9%，主要由大量 0 构成，量化明显。
- `pos` 和 action 完整；本数据中大角度约为开放，小角度约为闭合。

### H1 — contact-transition information

`load` 在粗阶段上呈现强而可重复的符号结构：

- free/open → closing/contact：episode 中位差 `+92`，60/60 同方向；帧级 Cohen's d=`+1.61`，分布重叠 `0.426`。
- free/open → release：episode 中位差 `-110`，60/60 同方向；d=`-1.87`，重叠 `0.279`。

这支持 `load` 含有闭合/释放转换信息，但正负号同时强烈跟随运动方向，因此不能把它直接解释为接触力。

### H2 — stable grasp / suspended hold information

不支持“单帧 load/current 可稳定识别 Hold”：

- stable grasp 的 `load` 相对 free/open：d=`0.069`，重叠 `0.801`，跨 episode 主方向仅约 52%。
- lift/place 的 `load`：d=`-0.037`，重叠 `0.733`。
- stable grasp / lift-place 的 `current`：d=`-0.212/-0.199`，重叠均约 `0.93`。

相比之下 `gripper.pos` 几乎完全分离 free 与稳定闭合状态，但这只是几何位置证据，不是 force-related signal 的额外贡献。

### H3 — temporal information

`load` 的窗口特征明显优于阶段单值：

- stable grasp：单值 d=`-0.028`、重叠 `0.650`；15 帧窗口最大值 d=`-1.492`、重叠 `0.367`。
- lift/place 尾部：单值 d=`-0.355`、重叠 `0.633`；15 帧均值 d=`-5.442`、重叠 `0.017`。

第二项不能单独解释为 suspended hold，因为 `lift_place` 同时混合抬升、搬运、重新支撑及释放前过渡。它只能证明时间上下文有信息，不能证明已识别真实 Hold 或 support-regained。

`current` 的最佳窗口特征仍弱（stable grasp 最大 |d|=`0.725`；lift/place 最大 |d|=`0.305`），不宜作为当前主信号。

### H4 — expert gripper stability

expert action 在合并的 lift/place 区间并非严格恒定：

- action range 中位数 `5.27°`（P10–P90 `3.92–6.99°`）；
- 从区间初始闭合命令到最大重新张开量中位数 `4.11°`（P10–P90 `1.68–6.01°`）；
- 15/60 episodes（25%）重新张开至少 `5°`。`5°` 仅为审计描述阈值，不是事前 PASS Gate。

因此真机“Lift 后重新张爪”不能全部归因于 policy prediction；旧 demonstration 本身已经提供逐渐回开的监督。由于 S4/S5 未细分，也不能断言这些变化发生在纯悬空 Hold，而非接近放置/重新支撑阶段。

## Mandatory Gates

| Gate | Status | Result |
|---|---|---|
| G0 Provenance | PASS | 原始 Formal3、60 episodes、schema 与哈希已冻结；MASTER/NOMASTER 关系已说明 |
| G1 Signal Availability | PASS | 四信号逐帧完整；`current` 低分辨率异常已报告 |
| G2 Phase Recoverability | PASS with limitation | 人工边界支持五个粗阶段；S1/S2、S4/S5 为 unresolved |
| G3 Single-Episode Structure | PASS | ep13 四信号时序图已生成 |
| G4 Cross-Episode Consistency | PASS | 60 集对齐、分布、效应量和符号一致性已生成 |
| G5 Temporal Separability | PASS | 15 帧窗口与单值完成比较；仅 `load` 明显获益 |
| G6 Hold Action Quality | PASS | 60 集 Hold/Lift action 稳定性已统计，发现系统性渐开监督 |
| G7 Real Robot | NA | decision 明确禁止真机 |

## Supported and unsupported claims

可以支持：

> 历史数据中的 `gripper.load` 含有跨 episode 可重复的闭合/释放转换信息；时间窗口比单帧绝对值更适合后续最小 Force Discovery。旧 expert action 在 lift/place 中存在逐渐回开，可能是现有 policy 真机重新张爪的一部分数据来源。

不能支持：

- `load` 是经过物理单位标定的真实夹持力；
- `current` 对 stable hold 有可靠区分能力；
- 已自动识别 contact、suspended hold 或 support-regained；
- 已实现 force-aware policy 或闭环 force control；
- 能从旧数据估计真机成功率。

## Limitations / anomalies

- 数据没有 episode success/failure 元数据，不能完成真实“成功 vs 失败”分组；未用异常 episode 冒充失败样本。
- 旧数据只有粗视频审核边界，S2 contact、S4 suspended、S5 support-regained 缺少独立 ground truth。
- `load` 明显受运动方向影响；任何后续模块必须同时输入 pos/action history 或显式去除运动瞬态混淆。
- `current` 大量为 0 且只有 18 个离散值。

## Evidence

- 紧凑指标：`metrics.json`
- Gate 审计：`validation.json`
- 图：`figures/`
- Canonical raw evidence：`results/experiments/F0_LEGACY_FORCE_CALIBRATION/`

本轮完成后停止；未进入 F1。
