# P4.8-r5 complete single-task pilot

**G2_PROTOTYPE_PASS / G3_TRAINING_PASS / G4_CLOSED_LOOP_PASS**. G4 means all scheduled rollouts executed; it does not require a nonzero success count.

| Model | Updates | Simulator successes | Dev9 h0 arm raw MAE | Dev9 h0 gripper raw MAE |
|---|---:|---:|---:|---:|
| act | 10000 | 4/10 | 0.080250 | 0.213898 |
| legacy | 10000 | 4/10 | 0.078747 | 0.210692 |
| fixed | 10000 | 4/10 | 0.078671 | 0.217038 |

Raw action errors are OSC controller-input units, not SO101 joint degrees or measured end-effector motion. Dev9 is teacher-forced internal diagnosis, separate from simulator success.

All three jobs use the fixed Train35/Dev9 split and one seed. P1 Legacy/Fixed share initial weights, prototype bytes, data order and budget; only online DINO normalization differs. ACT has the native architecture/loss/optimizer defaults with disclosed interface,10k/seed1/sampler/cache/numerical-mode overrides.

The prototype method matches formal6: four fixed Train frames, four Codex-selected target-interior points, scaled35px ellipse/patch-center sampling, frozen single-normalized DINO and mean/L2. This contains a small executor-supplied target prior; it is not unsupervised. No old cup coordinates or bank were used.

Dataset RGB already follows the converted orientation; no training flip. Raw simulator RGB is flipped once by the official processor. Public conversion code explains10FPS metadata; native one-action/one20Hz-step execution is frozen without interpolation or repetition. Exact historical publisher conversion commit remains unknown.

Object ON/OFF uses the same trained weights and is a mechanism diagnostic, not a fresh training ablation. Failure-stage labels remain UNKNOWN when only the simulator success flag is available; videos and traces are retained.

All checkpoints, optimizer/RNG states, raw logs, trajectories, videos and engineering smoke evidence remain on the server. Compact ZIP excludes large caches/checkpoints and contains unique paths and SHA pointers. No Final15, hardware operation, extra seed/task, HF/Git publication or automatic full-suite expansion.

Independent Builder Review and Red-Team Review are PENDING. The executor checks below do not substitute for these reviews.
