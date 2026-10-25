# R5.10 A轮真机控制 CSV 归档（2026-10-25）

`outputs/r5_10/session_01/` 全部试次的 `R5_10_M*_A_01.csv` 字节级归档到本目录。文件名用试次目录名（含 `_prev_<时间戳>`）区分同模型的多次尝试；`manifest.json` 记录每个 CSV 的 SHA256。

状态是程序记录的事实（`summary.json`），不推断成败；阶段 P0–P6 全部 `UNREVIEWED`，`valid_initial=null`（人工复位、无数值 Home/容差核验）。视频/首帧/configuration/stage_review 仍留在本地 `outputs/`（gitignore）。

| 试次目录 | 模型 | 配方 | 状态 | 行数 | 有CSV |
|---|---|---|---|---|---|
| R5_10_M1_A_01 | M1 | C10-S3-80k | SAFETY_ABORT | 284 | yes |
| R5_10_M1_A_01_prev_20261025_190423 | M1 | C10-S3-80k | ERROR(相机) | 110 | yes |
| R5_10_M1_A_01_prev_20261025_190518 | M1 | C10-S3-80k | SAFETY_ABORT | 604 | yes |
| R5_10_M1_A_01_prev_20261025_190605 | M1 | C10-S3-80k | SAFETY_ABORT | 832 | yes |
| R5_10_M1_A_01_prev_20261025_190627 | M1 | C10-S3-80k | ERROR(相机) | 33 | yes |
| R5_10_M1_A_01_prev_20261025_190641 | M1 | C10-S3-80k | SAFETY_ABORT | 0 | yes |
| R5_10_M1_A_01_prev_20261025_190723 | M1 | C10-S3-80k | SAFETY_ABORT | 645 | yes |
| R5_10_M1_A_01_prev_20261025_190756 | M1 | C10-S3-80k | SAFETY_ABORT | 399 | yes |
| R5_10_M1_A_01_prev_20261025_190842 | M1 | C10-S3-80k | SAFETY_ABORT | 711 | yes |
| R5_10_M1_A_01_prev_20261025_190926 | M1 | C10-S3-80k | SAFETY_ABORT | 727 | yes |
| R5_10_M1_A_01_prev_20261025_191927 | M1 | C10-S3-80k | SAFETY_ABORT | 890 | yes |
| R5_10_M1_A_01_prev_20261025_191955 | M1 | C10-S3-80k | SAFETY_ABORT | 369 | yes |
| R5_10_M1_A_01_prev_20261025_192022 | M1 | C10-S3-80k | SAFETY_ABORT | 334 | yes |
| R5_10_M1_A_01_prev_20261025_192051 | M1 | C10-S3-80k | SAFETY_ABORT | 307 | yes |
| R5_10_M1_A_01_prev_20261025_192150 | M1 | C10-S3-80k | SAFETY_ABORT | 1323 | yes |
| R5_10_M1_A_01_prev_20261025_192539 | M1 | C10-S3-80k | SAFETY_ABORT | 308 | yes |
| R5_10_M1_A_01_prev_20261025_192708 | M1 | C10-S3-80k | TIME_LIMIT | 1782 | yes |
| R5_10_M1_A_01_prev_20261025_192902 | M1 | C10-S3-80k | SAFETY_ABORT | 573 | yes |
| R5_10_M1_A_01_prev_20261025_192946 | M1 | C10-S3-80k | SAFETY_ABORT | 503 | yes |
| R5_10_M1_A_01_prev_20261025_193056 | M1 | C10-S3-80k | SAFETY_ABORT | 1434 | yes |
| R5_10_M1_A_01_prev_20261025_193226 | M1 | C10-S3-80k | SAFETY_ABORT | 476 | yes |
| R5_10_M1_A_01_prev_20261025_193253 | M1 | C10-S3-80k | SAFETY_ABORT | 296 | yes |
| R5_10_M1_A_01_prev_20261025_193344 | M1 | C10-S3-80k | SAFETY_ABORT | 1049 | yes |
| R5_10_M1_A_01_prev_20261025_193411 | M1 | C10-S3-80k | SAFETY_ABORT | 351 | yes |
| R5_10_M1_A_01_prev_20261025_193639 | M1 | C10-S3-80k | SAFETY_ABORT | 1548 | yes |
| R5_10_M1_A_01_prev_20261025_193723 | M1 | C10-S3-80k | SAFETY_ABORT | 730 | yes |
| R5_10_M2_A_01 | M2 | C5-S2-80k | SAFETY_ABORT | 290 | yes |
| R5_10_M2_A_01_prev_20261025_191329 | M2 | C5-S2-80k | ERROR(相机) | 94 | yes |
| R5_10_M2_A_01_prev_20261025_191406 | M2 | C5-S2-80k | ERROR(相机) | 56 | yes |
| R5_10_M2_A_01_prev_20261025_191449 | M2 | C5-S2-80k | SAFETY_ABORT | 528 | yes |
| R5_10_M2_A_01_prev_20261025_191531 | M2 | C5-S2-80k | SAFETY_ABORT | 779 | yes |
| R5_10_M2_A_01_prev_20261025_193127 | M2 | C5-S2-80k | SAFETY_ABORT | 413 | yes |
| R5_10_M3_A_01 | M3 | C5-S2-SWA_60_80 | SAFETY_ABORT | 309 | yes |
| R5_10_M3_A_01_prev_20261025_191123 | M3 | C5-S2-SWA_60_80 | SAFETY_ABORT | 597 | yes |
| R5_10_M3_A_01_prev_20261025_191156 | M3 | C5-S2-SWA_60_80 | SAFETY_ABORT | 484 | yes |
| R5_10_M3_A_01_prev_20261025_193156 | M3 | C5-S2-SWA_60_80 | SAFETY_ABORT | 524 | yes |
| R5_10_M4_A_01 | M4 | C5-S3-SWA_60_80 | SAFETY_ABORT | 934 | yes |
| R5_10_M4_A_01_prev_20261025_191648 | M4 | C5-S3-SWA_60_80 | SAFETY_ABORT | 576 | yes |
| R5_10_M4_A_01_prev_20261025_191811 | M4 | C5-S3-SWA_60_80 | SAFETY_ABORT | 384 | yes |
| R5_10_M4_A_01_prev_20261025_191833 | M4 | C5-S3-SWA_60_80 | ERROR(标定) | 0 | no |

**状态说明**：`SAFETY_ABORT`=操作员 Ctrl+C/Esc 手动中止（多数），不等于任务失败；`TIME_LIMIT`=跑满 60s 未中止；`ERROR(相机)`=OpenCVCamera(4) read thread not running（设备未释放/交接）；`ERROR(标定)`=电机 `Max_Position_Limit` no status packet。阶段成败不在此表，需视频审阅。
