# LeRobot 半满透明塑料杯数据采集细节说明

> 适用对象：本项目新一版 60-episode 正式数据采集
> 目标：构建一个具有真实局部位置变化、动作阶段清晰、Hold supervision 稳定、并完整保留 gripper `pos/load/current` 的数据集，用于后续 ACT Baseline、Position、Force 与 Integration 实验。
> 本说明用于实际采集操作；科研授权与 PASS/FAIL 仍以对应 `decision.md` 为准。

---

# 1. 采集目标

本次数据采集不再采用旧的 Left / Center / Right 三个离散位置。

新的采集场景为：

> **一个主抓取区域 + 局部自然位置扰动。**

目标不是训练模型做大范围空间泛化，而是让策略能够在一个稳定、真实的局部工作区域内，根据杯子的实际位置进行正确的 Reach 和 Alignment，同时为后续 Force / Contact 分析保留清晰的时序数据。

完整任务流程：

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

# 2. 是否可以继续使用原来的采集脚本

可以。

如果原有 LeRobot 采集脚本能够正常记录以下信息，则不需要重写采集系统：

- Top RGB；
- Wrist RGB；
- arm joint positions；
- `gripper.pos`；
- `gripper.load`；
- `gripper.current`；
- action；
- frame index / timestamp。

本次真正需要改变的是：场景布置、杯子摆放方式、采集位置分布、遥操作动作节奏、Hold 阶段夹爪语义、采集后的 QC。

如果原脚本缺少 `load/current` 或当前 observation/action schema 与项目正式 schema 不一致，则在正式采集前修正。

# 3. 场景布置

## 3.1 Nominal Grasp Region

选择一个机器人最容易稳定抓取的位置作为 nominal center `P0`。

第一版建议局部区域：

```text
横向：约 ±20 mm
纵向：约 ±15 mm
```

即整体约：

```text
40 mm × 30 mm
```

示意：

```text
                 后

        ┌──────────────────┐
        │                  │
        │        P0        │
        │                  │
        └──────────────────┘

                 前
```

40 × 30 mm 是采集设计范围，不是必须画出的视觉目标框。最终论文应报告实际采集到的杯中心分布范围。

## 3.2 P0 的选择原则

P0 应满足：

- 机器人各关节处于舒适工作区；
- 左右 ±20 mm 均可安全到达；
- 前后 ±15 mm 均可安全到达；
- 抬升约 60 mm 不会碰桌面、相机或机械结构；
- Top camera 能完整看到杯子及整个局部区域；
- Wrist camera 在接近杯子时不会严重遮挡；
- 抓取高度附近不存在明显关节极限或奇异位形。

优先选择机器人最容易稳定、重复完成完整抓放的 nominal point。

# 4. 区域标记

推荐：

- 4 个小型、低反光角标；
- 每个约 3–5 mm；
- 或把辅助边界放在相机 crop 之外。

不推荐：

- 黑色高对比完整矩形；
- 红色/蓝色显眼胶带；
- 明显反光透明胶带；
- 把 4×3 内部网格画在桌面上。

原因：明显边框可能成为视觉 shortcut，模型学“框的位置”而不是“杯子的位置”。

# 5. 杯子与环境要求

正式数据集第一版尽量固定：

- 同型号透明塑料杯；
- 近似半杯水；
- 同一类液体；
- 杯体基本竖直；
- 不新增人工视觉标记；
- 桌面不变；
- camera placement 不变；
- robot base 不变；
- 光照尽量稳定；
- Home pose 不变；
- nominal grasp height 不变；
- nominal lift height 不变。

第一版暂不主动引入多杯型、大范围水量变化、大范围光照变化、相机移动、大范围杯姿变化和大范围工作空间变化。

# 6. 4×3 空间分层

将约 40 × 30 mm 区域在采集管理上划分为：

```text
4 列 × 3 行 = 12 个 spatial strata
```

示意：

```text
        左                      右

        A1    A2    A3    A4
        B1    B2    B3    B4
        C1    C2    C3    C4
```

如果总范围约 40 × 30 mm，则每个 stratum 约 10 × 10 mm。

这些区域只用于人工采集管理，不是 12 个固定抓取点。

# 7. 60 条 episode 的分配

每个 stratum 采 5 条，总计：

```text
12 × 5 = 60 episodes
```

每个 stratum 内部都要有小范围自然位置变化。不要全部放在格子中心。

目标是形成“一个连续局部区域中的 60 个自然位置样本”，而不是“12 个固定坐标各重复 5 次”。

# 8. 采集 Round

推荐：

```text
Round 1：12 个 strata 各 1 条
Round 2：12 个 strata 各 1 条
Round 3：12 个 strata 各 1 条
Round 4：12 个 strata 各 1 条
Round 5：12 个 strata 各 1 条
```

每一 Round 内随机打乱 12 个 strata 的顺序。

禁止 A1 连采 5 条后再 A2 连采 5 条，以减少操作者习惯、疲劳、时间漂移和光照变化与空间位置绑定。

# 9. 每条 episode 开始前

每次采集前：

1. 机器人回统一 Home；
2. gripper 保持 OPEN；
3. 根据采集计划把杯子摆入目标 stratum；
4. 在该 stratum 内稍微随机改变具体位置；
5. 保持杯体竖直；
6. 水量约半杯；
7. 确认 Top / Wrist camera 正常；
8. 确认线缆无明显干涉；
9. 开始正式录制。

不要为了轨迹漂亮而每次都把杯子摆到完全相同的位置。

# 10. 每条 episode 的操作流程

## S0 — Home + Open

- 固定 Home；
- gripper 完全 OPEN；
- 稳定约 0.3–0.5 s。

## S1 — Reach

- 从 Home 向杯子运动；
- gripper 全程 OPEN；
- 杯子偏左/右/前/后时，从较早阶段就自然产生对应轨迹调整；
- 禁止所有轨迹先走 P0，再最后几厘米突然修正。

## S2 — Fine Alignment

- 进入杯子附近最后约 30–50 mm 后减速精确对准；
- gripper 仍保持 OPEN；
- 不开始 closing；
- 不与 Lift 重叠。

## S3 — Pre-Grasp Stabilization

- 达到最终抓取位置；
- arm 基本静止；
- gripper OPEN；
- 保持约 0.2–0.3 s。

## S4 — Grasp

- 开始闭合 gripper；
- 其他关节尽量保持静止；
- 不边闭爪边抬升；
- 建立有效杯壁接触。

## S5 — Grasp Stabilization

- 闭爪完成后保持约 0.5–0.8 s，推荐约 0.6 s；
- gripper 稳定闭合；
- arm 基本不动；
- 不开始 Lift；
- 不重新张开。

## S6 — Lift

- 近似垂直抬升；
- 第一版目标高度约 60 mm；
- 推荐 0.6–1.0 s 完成；
- Lift 过程中 gripper target 基本保持稳定。

## S7 — Hold

- 抬到目标高度后悬停约 1.0 s；
- arm 尽量稳定；
- gripper 保持稳定闭合；
- 不做 release；
- 不人为减小夹持程度。

## S8 — Lower

- 稳定把杯子放回原位置附近；
- gripper 继续保持稳定抓持；
- 不边下降边打开。

## S9 — Place Stabilization

- 杯底重新接触桌面后，不马上松手；
- 继续保持 gripper 闭合约 0.3–0.5 s，推荐约 0.4 s。

## S10 — Release

- Place Stabilization 结束后；
- arm 尽量保持静止；
- 打开 gripper；
- 不同时执行明显 Retreat。

## S11 — Release Stabilization

- gripper 打开后原地保持约 0.3–0.5 s，推荐约 0.4 s。

## S12 — Retreat

- gripper 保持 OPEN；
- 沿安全方向退出约 50–80 mm。

## S13 — Return Home

- 回到固定 Home；
- gripper 保持 OPEN；
- 稳定约 0.3–0.5 s；
- episode 结束。

# 11. 操作口诀

```text
开着走
→ 到位停
→ 闭
→ 闭着停
→ 抬
→ 停
→ 放
→ 放稳停
→ 开
→ 开着停
→ 退
→ 回 Home
```

核心：关键动作语义不要重叠。

# 12. Gripper 操作核心要求

新数据必须修正旧数据中 Lift / Hold 阶段 gripper target 逐渐回开的问题。

目标模式：

```text
OPEN ────────────────╲
                      CLOSED ─────────────────────────
                                                    ╲
                                                     OPEN ──────
```

即：

- Home / Reach / Fine Alignment = OPEN
- Grasp = Closing
- Grasp Stabilization → Lift → Hold → Lower → Place Stabilization = 基本稳定 CLOSED
- Release = Opening
- Retreat / Home = OPEN

# 13. 抓取高度

推荐夹取杯体中部附近，约为杯高 45%–60%。

避免太靠杯沿、太靠杯底或太接近桌面。正式采集期间抓取高度尽量保持一致。

# 14. Lift 高度

第一版推荐约 60 mm。

目标是明确离开桌面并形成 suspended hold，同时保持安全、减少无必要洒水风险。

如果 60 mm 在真实空间不合理，应在正式采集前调整并固定，不要采到一半再改变。

# 15. 失败与异常 episode

每条至少分类：

```text
VALID
HARDWARE_INVALID
OPERATOR_INVALID
TASK_FAILURE
```

## HARDWARE_INVALID

如 camera 丢帧、motor communication failure、文件损坏、telemetry 缺失、录制截断。

处理：保留记录并重采对应位置。

## OPERATOR_INVALID

如忘记 Hold、提前 Release、杯子摆错 stratum、Lift 时误开夹爪、严重违反阶段操作要求。

处理：标记 invalid，并重采对应位置。

## TASK_FAILURE

如 Reach 正常但没抓准、杯子滑落、Lift 失败、Hold 掉杯、明显洒水。

处理：保留原始数据，不得因为失败自动删除。是否进入最终训练数据由后续 Dataset Review 决定。

# 16. 采集记录表建议

建议维护：

| Round | Stratum | Episode | Position Note | Validity | Reach | Grasp | Lift | Hold | Place | Release | Drop/Spill | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

该表仅用于采集管理和后续 QC，不作为模型输入。

# 17. Force-related signals

正式采集必须继续保留：

```text
gripper.pos
gripper.load
gripper.current
gripper action / target
```

当前解释：

- `gripper.load`：force-related / load proxy；
- `gripper.current`：辅助电机状态信号；
- `gripper.pos`：夹爪几何状态；
- action / target：expert intention。

采集阶段不要把 `load` 直接解释成真实夹持力。

# 18. 采完 60 条后不要马上训练

完整流程：

```text
60 episodes 采集完成
↓
Dataset QC
↓
Spatial Coverage Review
↓
Motion Phase Review
↓
Hold Supervision Review
↓
Force Signal Integrity Review
↓
用户 + ChatGPT 验收
↓
Dataset Freeze
↓
Split Freeze
↓
ACT Baseline
```

禁止采完马上拿全部数据训练。

# 19. 采集后重点 QC

## Spatial

检查实际杯中心 `(cx, cy)` 分布：

- 是否有真实局部位置变化；
- 是否严重集中在 P0；
- 12 个 strata 是否都有覆盖。

## Motion

检查：

- Reach 与 Grasp 是否分开；
- Grasp 与 Lift 是否分开；
- Hold 是否存在；
- Lower 与 Release 是否分开。

## Gripper

重点检查 Grasp Stabilization → Lift → Hold → Lower → Place Stabilization 期间 gripper target 是否基本稳定。

如果仍大量逐渐 reopen，新数据集不能直接冻结。

## Force Signals

检查 `pos/load/current/action` 是否完整、同步并具有清晰时间结构。

# 20. 核心原则总结

本次采集最重要的不是“60 条都非常漂亮”，而是同时满足：

1. 位置有真实的小范围变化；
2. Reach trajectory 会随杯位自然变化；
3. 关键动作阶段清晰，不要重叠；
4. Grasp 后到 Place 前夹爪保持稳定抓持；
5. 完整记录 pos/load/current/action；
6. 任务失败不等于数据无效；
7. 正式训练前先做 Dataset QC 和 Split Freeze。

一句话：

> **任务语义保持一致，杯位自然变化，轨迹根据杯位变化，Hold supervision 清晰，force-related signals 完整。**
