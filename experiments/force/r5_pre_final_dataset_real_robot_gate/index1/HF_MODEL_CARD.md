---
license: apache-2.0
tags:
- robotics
- lerobot
- act
- so101
---

# R5 pre-final-dataset SO101 real-robot gate

Five frozen inference routes for a transparent, water-filled cup task. This is an experimental deployment bundle, not an independently validated robot controller or a claim of success.

| ID | Global checkpoint | Local checkpoint |
| --- | --- | --- |
| M1 | Strong-G Train36 seed3 / 80k | none |
| M2 | Strong-G Full60 seed3 / 80k | none |
| M3 | Same global as M1 | F4.2 seed3 / 7k |
| M4 | Same global as M2 | Full60 seed3 / 7k |
| M5 | Strong-G Full60 seed4 / 80k | none |

M5 was selected before robot testing using the R4 offline combined arm/gripper and spatial diagnostics. No model is selected using R5 robot outcomes. Every route retains its own Train36/Full60 normalizer. Local residual limit is unchanged at ±4.832699459791184 native gripper units; local affects only the gripper after global temporal ensemble/denormalization and before the existing robot limit.

Download `R5_deploy_bundle.tar.gz`, `R5_project_files.tar.gz`, and `checksums.json`; follow `REAL_ROBOT_TEST_GUIDE.md`. The deploy archive extracts to `R5/` and includes weights, normalization, prototypes, and offline vision caches. Full training videos/dataset are not required for inference. The project-files archive contains source and guides, not credentials.

Project: https://github.com/QianYeme/lerobot/tree/det

Use `control_r5.py`, not the older F5 controller. R4 requires `cudnn.benchmark=False`; TF32 remains enabled. R5 input load arithmetic explicitly matches R4 cached float32 normalization. CLI temporal ensembling and relative target limit remain editable but must be fixed across this controlled study.

The planned study is five models × familiar A/B/C positions (15 trials), blocked rotation, no rescue repetitions. Primary observations are target-directed approach and valid grasp geometry. Human assistance is recorded separately, never counted as autonomous success. Physical limits and emergency stop remain required.

`runtime_smoke.json` is server GPU real-RGB batch1 first/reset and local-composition evidence only, not target-host camera/serial integration or closed-loop safety. Robot trials at publication: **0**. `R5_COMPLETE=false`. No new training, final-dataset collection or robot actuation is performed by the upload process.
