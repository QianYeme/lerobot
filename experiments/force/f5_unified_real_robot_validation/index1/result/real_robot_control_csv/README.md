# F5 真机控制 CSV 汇总（session_01，2026-10-22）

来源：`outputs/f5/session_01/`（`outputs/` 在 `.gitignore` 内，故把 CSV 复制到此处随仓库保存）。
命名规则：`<trial 目录名>.csv`；`_prev_<时间戳>` 表示同一试次被后一次运行取代的旧尝试，**保留不删除**。

- 全部尝试均为 A 点位。
- 20 份 CSV；另有 3 次启动即失败的尝试（相机/连接错误，0 步）没有生成 CSV。
- 所有尝试的 `summary.status` 都是 `SAFETY_ABORT`（操作员 Ctrl+C）或 `ERROR`，**没有一次达到 60 秒自然结束**。
- 所有 `stage_review.json` 仍为 `UNREVIEWED`：**本表不含任何成功/失败判定**，阶段结果需人工按视频填写后才能得出。

## 当前（每个模型最新一次）

| 试次 | 模型 | 状态 | 步数 |
|---|---|---|---|
| `F5_P1Hist_A_01` | P1-Hist | SAFETY_ABORT | 580 |
| `F5_StrongG_A_01` | Strong-G | SAFETY_ABORT | 811 |
| `F5_LLoad_A_01` | L-Load | SAFETY_ABORT | 287 |
| `F5_NewGL_A_01` | New-GL | SAFETY_ABORT | 507 |

## 全部尝试

| CSV | 模型 | 状态 | 步数 | 类别 |
|---|---|---|---|---|
| `F5_P1Hist_A_01.csv` | P1-Hist | SAFETY_ABORT | 580 | LATEST |
| `F5_P1Hist_A_01_prev_20261022_170626.csv` | P1-Hist | SAFETY_ABORT | 522 | SUPERSEDED |
| `F5_P1Hist_A_01_prev_20261022_170720.csv` | P1-Hist | SAFETY_ABORT | 762 | SUPERSEDED |
| `F5_P1Hist_A_01_prev_20261022_170851.csv` | P1-Hist | SAFETY_ABORT | 650 | SUPERSEDED |
| `F5_StrongG_A_01.csv` | Strong-G | SAFETY_ABORT | 811 | LATEST |
| `F5_StrongG_A_01_prev_20261022_170929.csv` | Strong-G | SAFETY_ABORT | 1272 | SUPERSEDED |
| `F5_StrongG_A_01_prev_20261022_171023.csv` | Strong-G | SAFETY_ABORT | 1023 | SUPERSEDED |
| `F5_StrongG_A_01_prev_20261022_171112.csv` | Strong-G | SAFETY_ABORT | 516 | SUPERSEDED |
| `F5_LLoad_A_01.csv` | L-Load | SAFETY_ABORT | 287 | LATEST |
| `F5_LLoad_A_01_prev_20261022_170011.csv` | L-Load | SAFETY_ABORT | 309 | SUPERSEDED |
| `F5_LLoad_A_01_prev_20261022_171158.csv` | L-Load | SAFETY_ABORT | 1365 | SUPERSEDED |
| `F5_LLoad_A_01_prev_20261022_171229.csv` | L-Load | SAFETY_ABORT | 489 | SUPERSEDED |
| `F5_LLoad_A_01_prev_20261022_171252.csv` | L-Load | SAFETY_ABORT | 306 | SUPERSEDED |
| `F5_NewGL_A_01.csv` | New-GL | SAFETY_ABORT | 507 | LATEST |
| `F5_NewGL_A_01_prev_20261022_170237.csv` | New-GL | SAFETY_ABORT | 254 | SUPERSEDED |
| `F5_NewGL_A_01_prev_20261022_170339.csv` | New-GL | SAFETY_ABORT | 1455 | SUPERSEDED |
| `F5_NewGL_A_01_prev_20261022_170408.csv` | New-GL | SAFETY_ABORT | 452 | SUPERSEDED |
| `F5_NewGL_A_01_prev_20261022_171328.csv` | New-GL | SAFETY_ABORT | 387 | SUPERSEDED |
| `F5_NewGL_A_01_prev_20261022_171408.csv` | New-GL | SAFETY_ABORT | 523 | SUPERSEDED |
| `F5_NewGL_A_01_prev_20261022_171459.csv` | New-GL | SAFETY_ABORT | 938 | SUPERSEDED |

`manifest.json` 为机器可读的同表（含字节数）。

## 未包含

- 视频、首帧 jpg、`summary.json`、`stage_review.json`、`local_regulator_metrics.json` 仍在 `outputs/f5/session_01/` 本地保留，未纳入 Git。
- 没有 C 轮/B 轮；12 次正式预算只执行了 A 轮的 4 个模型（且每个模型多次重试）。
- `F5_COMPLETE=false`。
