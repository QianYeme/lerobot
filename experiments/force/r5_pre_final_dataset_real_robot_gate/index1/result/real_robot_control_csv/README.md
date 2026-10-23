# R5 A轮真机控制 CSV 归档（2026-10-23）

`outputs/r5/session_01/` 全部试次的 `R5_M*_A_01.csv` 字节级归档到本目录。文件名用试次目录名（含 `_prev_<时间戳>`），以区分同模型的多次尝试；`manifest.json` 记录每个 CSV 的 SHA256。

状态是程序记录的事实（`summary.json`），不推断成败；阶段 P0–P6 全部 UNREVIEWED，未做人工审核。

| 试次目录 | 模型 | 数据集 | 状态 | 行数 | 有CSV |
|---|---|---|---|---|---|
| R5_M1_A_01 | M1 | train36 | SAFETY_ABORT | 487 | yes |
| R5_M1_A_01_prev_20261023_225310 | M1 | train36 | CLEANUP_ERROR | 1023 | yes |
| R5_M1_A_01_prev_20261023_225410 | M1 | train36 | SAFETY_ABORT | 191 | yes |
| R5_M2_A_01 | M2 | full60 | SAFETY_ABORT | 140 | yes |
| R5_M2_A_01_prev_20261023_224041 | M2 | full60 | ERROR | 0 | no |
| R5_M2_A_01_prev_20261023_224205 | M2 | full60 | TIME_LIMIT | 1723 | yes |
| R5_M2_A_01_prev_20261023_225342 | M2 | full60 | SAFETY_ABORT | 315 | yes |
| R5_M3_A_01 | M3 | train36 | TIME_LIMIT | 748 | yes |
| R5_M3_A_01_prev_20261023_224246 | M3 | train36 | SAFETY_ABORT | 0 | no |
| R5_M3_A_01_prev_20261023_224401 | M3 | train36 | TIME_LIMIT | 1560 | yes |
| R5_M3_A_01_prev_20261023_224431 | M3 | train36 | SAFETY_ABORT | 440 | yes |
| R5_M3_A_01_prev_20261023_225030 | M3 | train36 | ERROR | 0 | no |
| R5_M3_A_01_prev_20261023_225153 | M3 | train36 | SAFETY_ABORT | 639 | yes |
| R5_M4_A_01 | M4 | full60 | SAFETY_ABORT | 202 | yes |
| R5_M4_A_01_prev_20261023_224513 | M4 | full60 | ERROR | 3 | yes |
| R5_M4_A_01_prev_20261023_224545 | M4 | full60 | SAFETY_ABORT | 381 | yes |
| R5_M4_A_01_prev_20261023_224929 | M4 | full60 | SAFETY_ABORT | 1101 | yes |
| R5_M4_A_01_prev_20261023_225129 | M4 | full60 | SAFETY_ABORT | 815 | yes |
| R5_M5_A_01 | M5 | full60(seed4) | SAFETY_ABORT | 697 | yes |
| R5_M5_A_01_prev_20261023_224656 | M5 | full60(seed4) | ERROR | 0 | no |
| R5_M5_A_01_prev_20261023_224821 | M5 | full60(seed4) | ERROR | 35 | yes |

**四类非行为性中止**：`ERROR rows=0` 多为相机 `OpenCVCamera(4) read thread is not running` 或 `robot.connect(calibrate=False)` 失败（设备未释放/交接）；`CLEANUP_ERROR` 为 36.5s 正常跑完后清理阶段 motor bus `no status packet`；`TIME_LIMIT` 为跑满 60s 未抓取也未中止；`SAFETY_ABORT` 为操作员急停/中断。
