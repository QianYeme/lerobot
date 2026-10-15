# G1 Position Real-Robot Generalization

> D2.1身份复核：B3_ACT是历史名称，实际冻结模型为OC-ACT，模块在forward中确实执行；撤回纯ACT baseline表述。G1正式测试仍未完成。

> 2026-10-14 更新：以下硬件阻塞描述为准备阶段旧记录，已过期。首次真机定性测试已发生，正式 G1 尚未完成；当前交付见 `REAL_ROBOT_REVIEW_20261014.md`、`metrics.json`、`validation_review_20261014.json` 和 `proposal.md`。

`POSITION_REAL_ROBOT_GENERALIZATION_COMPLETE = FALSE`

Status: `BLOCKED_HARDWARE`. The frozen trial manifest is prepared, but no real-robot trial was run. Current workstation has the required RTX 4060 GPU but no SO101 serial port/cameras, and the frozen B3/P1/R3 checkpoints are not present locally.

Planned design: Phase A 3 nominal-center trials/model (9, excluded from primary rate), then Phase B 12 strata ? 2 trials ? 3 models (72 formal trials), total 81 planned trials. See `trial_manifest.json` and `preflight.md`.
