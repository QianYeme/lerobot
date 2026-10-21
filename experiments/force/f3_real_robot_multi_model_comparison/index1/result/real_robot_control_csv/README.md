# 真机控制 CSV 归档（control.csv）

来源：`outputs/f3_r1/session_01/<trial>/control.csv`，整合到本目录，文件名即 trial 目录名。
`_prev_*` 为被后续同名 trial 重命名保留的历史运行；无 `_prev_` 的为每模型/每次 shadow 的最终一次运行。

| trial | model | mode | status | rows | size |
|-------|-------|------|--------|-----:|-----:|
| R01_M0 | M0 | disabled | SAFETY_ABORT | 447 | 244908 |
| R01_M0_prev_20261021_112320 | M0 | disabled | SAFETY_ABORT | 674 | 367174 |
| R01_M0_prev_20261021_112842 | M0 | disabled | SAFETY_ABORT | 933 | 507896 |
| R01_M0_prev_20261021_114601 | M0 | disabled | SAFETY_ABORT | 607 | 331017 |
| R01_M0_prev_20261021_114703 | M0 | disabled | SAFETY_ABORT | 618 | 337159 |
| R01_M0_prev_20261021_114739 | M0 | disabled | SAFETY_ABORT | 427 | 233651 |
| R01_M1 | M1 | disabled | SAFETY_ABORT | 605 | 331465 |
| R01_M1_prev_20261021_114816 | M1 | disabled | SAFETY_ABORT | 1156 | 632171 |
| R01_M1_prev_20261021_114846 | M1 | disabled | SAFETY_ABORT | 276 | 151476 |
| R01_M2 | M2 | disabled | SAFETY_ABORT | 539 | 293722 |
| R01_M2_prev_20261021_112630 | M2 | disabled | SAFETY_ABORT | 1282 | 706294 |
| R01_M2_prev_20261021_114932 | M2 | disabled | TIME_LIMIT | 1781 | 973340 |
| R01_M2_prev_20261021_115005 | M2 | disabled | SAFETY_ABORT | 563 | 306828 |
| R01_M2_prev_20261021_115036 | M2 | disabled | SAFETY_ABORT | 238 | 131032 |
| R01_M3 | M3 | regulated | TIME_LIMIT | 1778 | 1065462 |
| R01_M3_prev_20261021_124158 | M3 | regulated | SAFETY_ABORT | 691 | 414657 |
| R01_M3_prev_20261021_125526 | M3 | regulated | SAFETY_ABORT | 369 | 222239 |
| SHADOW_01 | M3 | shadow | TIME_LIMIT | 1778 | 1065728 |
| SHADOW_01_prev_20261021_120629 | M3 | shadow | SAFETY_ABORT | 1345 | 819431 |
| SHADOW_02 | M3 | shadow | TIME_LIMIT | 1779 | 1085618 |
| SHADOW_03 | M3 | shadow | TIME_LIMIT | 1778 | 1075819 |
| SHADOW_03_prev_20261021_120431 | M3 | shadow | SAFETY_ABORT | 306 | 185249 |

列含义：step, t_s, wall_time, inference_s, processing_s, overrun, gripper.load, gripper.curr, predicted_delta, clipped_delta, hypothetical_gripper，以及 state/nominal/requested/sent 各关节字段。
