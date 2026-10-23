# R5 A 轮真机发现总结（2026-10-23）

范围：`outputs/r5/session_01/` 的 A 轮真实试次（仅 position=A）。**阶段 P0–P6 全部 UNREVIEWED，本文只记录程序事实 + 操作员现场观察，不推断成败率**。原始 CSV 已归档到 `real_robot_control_csv/`（含 SHA256 manifest）。

## 0. 试次概览

| 模型 | 数据集 | 尝试次数 | 状态（各次） |
|---|---|---|---|
| M1 Strong-G-36-S3 | train36 | 3 | SAFETY_ABORT / CLEANUP_ERROR(36.5s) / SAFETY_ABORT |
| M2 Strong-G-60-S3 | full60 | 4 | ERROR(connect) / TIME_LIMIT(60s) / SAFETY_ABORT / SAFETY_ABORT |
| M3 New-GL-36 | train36 | 6 | SAFETY_ABORT(0) / TIME_LIMIT(60s) / SAFETY_ABORT / ERROR(connect) / SAFETY_ABORT / TIME_LIMIT(60s) |
| M4 New-GL-60 | full60 | 5 | ERROR(3) / SAFETY_ABORT / SAFETY_ABORT / SAFETY_ABORT / SAFETY_ABORT |
| M5 G60-BadSeed-S4 | full60(seed4) | 3 | ERROR(connect) / ERROR(35) / SAFETY_ABORT |

共 21 次尝试、17 个非空 CSV。无一次 OPERATOR_END；`TIME_LIMIT` 是跑满 60s 未中止（未抓取），`SAFETY_ABORT` 是急停/中断，`ERROR rows=0` 是相机/串口连接失败。

## 1. 发现一：full60 模型不抓取（行为问题，不是供电问题）

**操作员现场观察**：三个 full60 模型（M2/M4/M5）出现共同问题——不抓取，甚至到不了塑料杯位置，只稍微向前伸一下就回到最终位置。train36 模型（M1/M3）能到达/抓取。

**关键判断：这不能用电池/混合模式解释。** 证据是 M2 有一次跑满 60 秒的试次（`R5_M2_A_01_prev_20261023_224205`，1723 行）**推理稳定在 ~24ms、overrun 仅 3.5%**，仍然不抓取。即供电正常、时序干净时 full60 依然失败，说明是策略本身的轨迹决策（full60 训练导致某种"向前试探后回撤"的失败模式），不是卡顿掩盖了抓取。

- 夹爪命令层面，五个模型都扫过相近的夹爪行程（requested gripper ~0→65~75），差异不在"开没开夹爪"，而在**空间轨迹是否到达杯子**——这需要视频/点位人工复核，CSV 本身不给杯位真值。
- 这是一条待查的模型级结论：36 集（train36）可达可抓，60 集（full60）反而不抓，方向与"数据越多越好"相反。可能与 60 集数据方差更大导致策略平均化/坍缩到安全回撤有关（与既有 ACTDet 抓取诊断的"模式坍缩"同族现象）。

## 2. 发现二：动作停顿（推理延迟尖峰，供电/混合模式导致）

**操作员现场观察**：推理时动作会停一段再动。当时开混合模式（Optimus），笔记本未插电。

**CSV 证据**（`inference_s`/`processing_s`/`overrun`/`wall_time` 间隔）呈明显双峰：

| 类型 | inf 均值 | inf p95 | inf 峰值 | overrun 占比 | 代表试次 |
|---|---|---|---|---|---|
| 正常 | ~0.020–0.035s | ≤0.042s | ≤0.29s | 3.5–8.9% | M2_224205、M3_224401、M4_224929… |
| 卡顿 | ~0.046–0.072s | 0.14–0.37s | 达 1.56s | 50–95% | M1 live、M3 live、M3_225153… |

卡顿试次里 `wall_time` 相邻行间隔最大到 1.5s——控制循环在推理上被阻塞，期间不发送新动作，从臂保持上一指令（看起来"停住"），恢复后才"再动"。这正是观察到的停顿。

**回答"会不会影响测试"**：
- **停顿（问题二）→ 会，且就是它造成的。** 未插电时 RTX 4060 Laptop 动态功耗被压、混合模式经核显中转，推理延迟从 ~24ms 恶化到 54–72ms 并周期性尖峰到 1.5s。这直接破坏 30Hz 平滑执行。
- **不抓取（问题一）→ 不会。** 供电只改变快慢/平滑，不改变策略"想往哪走"；且上述 M2 满 60s 平滑试次仍不抓，证明不抓取独立于供电。

## 3. 发现三：设备交接/释放错误（工程问题，非行为）

多次 `ERROR`/`CLEANUP_ERROR` 的 error.log 归为两类：
- `OpenCVCamera(4) read thread is not running`（相机 index4 未释放/未连接，重跑时交接失败）；
- `Failed to sync read 'Present_Position' ... no status packet`（电机总线掉包，多为跑完清理阶段或连接阶段）。

这是重跑时的设备释放/交接问题，不是模型行为。按既有指南，相机打开失败这类无效尝试保留日志、不悄悄删除。

## 4. 结论与建议

1. **full60 抓取失败是真实模型行为问题**（待查根因，优先：full60 权重/训练集/坍缩诊断），不是电池/混合模式的假象。
2. **停顿是供电/混合模式的工程问题**，已由推理延迟双峰佐证；重跑应**插电 + 独显直连（离散显卡模式）**，并可在真机指南/检查表里加一条"开测前确认插电 + 非混合模式 + nvidia-smi 正常"。
3. 设备交接错误（相机/串口不释放）建议在两次试次之间加固定释放与等待，或统一在同一进程内连跑。
4. 阶段 P0–P6 与 `validity` 仍需人工按视频逐条审核，`summarize_r5.py` 会在审核后产出 `per_trial_stage_metrics.csv` 等；当前所有 `stage_review.json` 为 UNREVIEWED/UNKNOWN，不得把终止状态当成败。

原始数据与归档：`outputs/r5/session_01/`（含视频/first-frame/logs）、`result/real_robot_control_csv/`（CSV 归档）、本文件。
