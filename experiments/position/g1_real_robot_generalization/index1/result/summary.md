# G1 Position Real-Robot Generalization

`POSITION_REAL_ROBOT_GENERALIZATION_COMPLETE = FALSE`

Status: `BLOCKED_HARDWARE`. The frozen trial manifest is prepared, but no real-robot trial was run. Current workstation has the required RTX 4060 GPU but no SO101 serial port/cameras, and the frozen B3/P1/R3 checkpoints are not present locally.

Planned design: Phase A 3 nominal-center trials/model (9, excluded from primary rate), then Phase B 12 strata ? 2 trials ? 3 models (72 formal trials), total 81 planned trials. See `trial_manifest.json` and `preflight.md`.
