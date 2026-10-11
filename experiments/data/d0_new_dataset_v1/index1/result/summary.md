# D0 New Dataset V1 — Interim Result

## Status

**BLOCKED_HARDWARE / READY_FOR_COLLECTION**。

本机未检测到机械臂串口或摄像头，且不存在本轮新采集的 LeRobot 数据集，因此无法真实完成60条采集及 D-G0–D-G5 QC。旧 `Fdataset/fk` 不符合本轮 provenance，未挪用。

## Completed engineering preparation

- 审计 `SOFollower`：`get_observation()` 会逐帧读取 `gripper.pos/load/curr`，schema 可满足 telemetry 合同。
- 修正 `lerobot_record.py`：原实现获得 `send_action()` 的限幅后返回值，却错误保存限幅前 action；现保存实际发送目标。
- 生成 seed=`20261008` 的冻结采集表：5 rounds × 12 strata，每轮每 stratum 恰好一次，共60条，五轮顺序不同。
- 表中预留位置估计、四类 episode status、Reach/Grasp/Lift/Hold/Place/Release/Drop/Spill 及阶段边界字段。
- 编写现场 runbook、smoke 命令模板与停止条件。
- 三项直接回归检查通过：覆盖性、确定性/轮次随机化、实际发送 action 保存语义。

## Not completed

- 真实硬件 smoke；
- 60 episodes 数据采集；
- 杯中心估计与空间覆盖；
- phase、Hold、force telemetry 和任务执行 QC；
- Dataset Review；
- split freeze（合同明确禁止在 QC PASS 前生成）。

## Evidence

- Runbook：`collection_runbook.md`
- Preflight metrics：`metrics.json`
- Gate 状态：`validation.json`
- 冻结采集表：`results/experiments/D0_NEW_DATASET_V1/collection_schedule.{csv,json}`

本状态不构成 D0 PASS。连接硬件后应从单 episode smoke 继续，不得直接开始60条正式采集。
