# decision_R5_PRE_FINAL_DATASET_REAL_ROBOT_GATE.md

## 0. Contract Header

- Experiment ID: `R5_PRE_FINAL_DATASET_REAL_ROBOT_GATE`
- Revision: `r0`
- Mode: `REAL_ROBOT_GATE_BEFORE_FINAL_DATASET`
- Status: `FROZEN / NOT_RUN`
- New training: `NO`
- New architecture: `NO`
- New dataset collection: `NO`
- Robot execution: `YES`
- Purpose:
  1. test whether Train36 -> Full60 produces a real-robot positioning gain;
  2. test whether offline global-seed differences transfer to real-robot behavior;
  3. decide whether the next large investment should be Final Dataset V1 (120–150 episodes) or spatial/perception diagnosis.

Recommended repository path:

```text
experiments/force/r5_pre_final_dataset_real_robot_gate/index1/decision.md
```

---

# 1. Why This Gate Exists

R4 has already established:

```text
R4_REPRODUCIBILITY_PASS = true
```

The old evaluation nondeterminism has been fixed.

Therefore:

- observed multi-seed differences are real training differences;
- Full60 seed3 showed improved spatial diagnostics;
- Full60 improvement was not consistent across all global seeds;
- Local MLP remains comparatively stable.

Before collecting 120–150 new demonstrations, test whether these offline differences matter on the robot.

---

# 2. Primary Questions

## Q1 — Data-scale transfer

Does:

```text
Strong-G-60 seed3
```

position the robot better than:

```text
Strong-G-36 seed3
```

under the same real-robot positions?

This is the cleanest Train36 -> Full60 test.

## Q2 — Full dual-role transfer

Does:

```text
New-GL-60
```

perform better than:

```text
New-GL-36
```

on positioning and downstream grasp execution?

## Q3 — Seed-to-robot transfer

Does a weaker Full60 global seed perform worse on the robot than the strong seed3 model?

This tests whether offline seed sensitivity has practical deployment meaning.

---

# 3. Frozen Model Set

## M1 — `Strong-G-36-S3`

Historical clean baseline:

```text
G-Load
Train36
seed3
80k
```

Role:
data-scale reference.

---

## M2 — `Strong-G-60-S3`

Full60 paired model:

```text
G-Load
Full60
seed3
80k
```

Role:
main Full60 global candidate.

---

## M3 — `New-GL-36`

Historical preferred dual-role model:

```text
global = Strong-G-36-S3
local = selected F4.2 local checkpoint
```

Role:
old dual-role reference.

---

## M4 — `New-GL-60`

Full60 dual-role candidate:

```text
global = Strong-G-60-S3
local = Full60 selected/frozen local seed3 / 7k
```

Role:
new main candidate.

---

## M5 — `G60-BadSeed`

Choose exactly one clearly weaker Full60 global seed based on the already reviewed offline package.

Preferred selection rule:

```text
use the seed with clearly worse combined arm/gripper and spatial diagnostics
```

Do NOT choose the weak seed after seeing robot behavior.

Record the exact seed before robot testing.

Role:
test whether offline global-seed weakness transfers to robot behavior.

---

# 4. Why Only One Bad Seed

Do not test all seed1/2/4.

The purpose is not a full robot seed study.

The purpose is to ask:

> Does a clearly weaker offline global seed also look weaker on the robot?

One weak-seed control is sufficient for this gate.

---

# 5. Fixed Robot Positions

Reuse the same known positions:

```text
Position A
Position B
Position C
```

Requirements:

- same workspace;
- same camera;
- same cup;
- same approximate fill state;
- same robot start-state procedure;
- no position expansion;
- no new jitter distribution in this gate.

This is a controlled comparison, not a new generalization benchmark.

---

# 6. Trial Budget

## Core four models

Each model runs once at A/B/C:

```text
Strong-G-36-S3 × 3
Strong-G-60-S3 × 3
New-GL-36 × 3
New-GL-60 × 3
```

Core total:

```text
12 trials
```

## Bad-seed control

Run:

```text
G60-BadSeed × 3
```

at A/B/C.

Extended total:

```text
15 trials
```

Do not add rescue repetitions.

---

# 7. Trial Order

Use blocked rotation to reduce order bias.

## Round A — Position A

```text
Strong-G-36-S3
Strong-G-60-S3
New-GL-36
New-GL-60
G60-BadSeed
```

## Round B — Position B

```text
Strong-G-60-S3
New-GL-36
New-GL-60
G60-BadSeed
Strong-G-36-S3
```

## Round C — Position C

```text
New-GL-36
New-GL-60
G60-BadSeed
Strong-G-36-S3
Strong-G-60-S3
```

Do not reorder after seeing results.

---

# 8. Minimal Pre-Run Check

Before each model family is used:

```text
checkpoint identity
config identity
input state schema
camera input
action dimensions
gripper sign
local residual bound for New-GL
cudnn.benchmark = False
```

The R4 reproducibility setting must be preserved for robot deployment.

No multi-layer shadow stage.

---

# 9. Safety

Keep only:

```text
correct checkpoint/config
existing hard action limits
existing residual limits
human emergency stop
stop on obviously dangerous behavior
```

No new safety architecture.

---

# 10. Primary Positioning Stages

This gate prioritizes target positioning.

Record:

```text
P0 target-directed approach
P1 valid grasp geometry
P2 close
P3 lift
P4 hold
P5 place
P6 release
```

Use:

```text
PASS
FAIL
UNKNOWN
```

---

# 11. Definition of P0

`P0 target-directed approach = PASS` if:

> The end-effector visibly moves toward the actual current cup location rather than toward a memorized/average central location.

This is the most important metric for Q1/Q3.

---

# 12. Definition of P1

`P1 valid grasp geometry = PASS` if:

> Before closure, the cup is in a plausible grasp region between/relative to the fingers.

P1 must be evaluated before any human assistance.

---

# 13. Core Robot Metrics

For every model report:

```text
P0 target-directed approach: x/3
P1 valid grasp geometry: x/3
P2 close: x/3
P3 lift: x/3
P4 hold: x/3
P5 place: x/3
P6 release: x/3
full autonomous success: x/3
```

Primary comparison metrics:

```text
P0
P1
```

Secondary:

```text
P3 lift
full autonomous success
```

---

# 14. Main Comparisons

## C1 — Data-scale global comparison

```text
Strong-G-36-S3
vs
Strong-G-60-S3
```

Interpretation target:

> Does Full60 improve real target-directed approach and grasp geometry?

## C2 — Data-scale dual-role comparison

```text
New-GL-36
vs
New-GL-60
```

Interpretation target:

> Does the Full60 gain survive after adding the Local MLP?

## C3 — Global vs dual-role under Full60

```text
Strong-G-60-S3
vs
New-GL-60
```

Interpretation target:

> Does Local MLP improve grasp/lift without hurting spatial positioning?

## C4 — Good seed vs bad seed

```text
Strong-G-60-S3
vs
G60-BadSeed
```

Interpretation target:

> Does offline global seed weakness transfer to robot behavior?

---

# 15. Assisted Recovery

Same rule as before.

If the autonomous policy is clearly unable to complete:

```text
autonomous phase ends
```

Then limited assistance may be used to inspect downstream behavior.

Record:

```text
first autonomous failure stage
assistance type
stages recovered afterward
```

Never count assisted completion as autonomous success.

---

# 16. Video / Runtime Log Handling

Video and CSV remain decoupled in time.

Only require shared identity:

```text
MODEL_ID
POSITION
TRIAL_ID
```

Video is used for:

```text
P0/P1 behavior
grasp geometry
lift
hold
place
release
assisted recovery
```

Runtime logs are used for:

```text
actions
load
gripper.pos
nominal gripper
delta gripper
final gripper
```

No forced video↔CSV timestamp alignment.

---

# 17. Local Metrics for New-GL Models

For:

```text
New-GL-36
New-GL-60
```

record:

```text
mean |delta|
max |delta|
near-bound fraction
sign-change rate
nonzero fraction
```

Check for:
- saturation;
- oscillation;
- obvious closure bias;
- release interference.

Do not attribute spatial approach differences to Local MLP because Local does not modify arm.

---

# 18. Outcome Categories

## `FULL60_ROBOT_SUPPORT`

Use if:

```text
Strong-G-60-S3 > Strong-G-36-S3
```

on P0/P1 descriptively, and:

```text
New-GL-60 >= New-GL-36
```

without introducing instability.

Interpretation:

> Increasing effective training data from 36 to 60 is supported by real-robot evidence.

This authorizes Final Dataset V1 collection.

---

## `FULL60_ROBOT_PARTIAL_SUPPORT`

Use if only one of the two paired comparisons improves.

Interpretation:

> More data helps, but interaction with architecture/local correction remains unresolved.

Final dataset collection may still proceed, but follow-up diagnosis must be preserved.

---

## `FULL60_ROBOT_NO_GAIN`

Use if Full60 does not improve P0/P1 for either family.

Interpretation:

> More of the same data alone is not yet supported as the primary remedy.

Do not immediately invest in a 120–150 episode collection.

Next step should emphasize spatial representation/policy mapping diagnosis.

---

## `SEED_EFFECT_TRANSFERS_TO_ROBOT`

Use as a secondary label if:

```text
Strong-G-60-S3
```

clearly outperforms:

```text
G60-BadSeed
```

on P0/P1 or action-chain behavior.

This supports treating seed stability as a practical deployment issue.

---

## `SEED_EFFECT_NOT_OBVIOUS_ON_ROBOT`

Use if the offline seed gap does not translate clearly to robot behavior.

Do not over-optimize offline seed metrics in that case.

---

# 19. Gate to Final Dataset V1

Proceed to Final Dataset V1 if either:

```text
FULL60_ROBOT_SUPPORT
```

or:

```text
FULL60_ROBOT_PARTIAL_SUPPORT
```

with a clear indication that denser spatial demonstrations remain useful.

If:

```text
FULL60_ROBOT_NO_GAIN
```

pause data collection and diagnose:

```text
object/spatial representation
position-action mapping
ACT averaging
camera/domain shift
```

first.

---

# 20. Final Dataset V1 — Next Planned Stage

If gate passes:

```text
target = 120–150 episodes
```

Recommended default:

```text
120 episodes
```

using approximately:

```text
15 position regions × 8 demonstrations
```

with:
- small position jitter;
- consistent task sequence;
- clear lift;
- hold ~0.8–1.0 s;
- consistent place/release.

This is not executed in R5.

---

# 21. No Post-Hoc Changes

During R5 do not:

```text
retrain
change checkpoint
change seed
change residual bound
change camera
change position set
change task
change cup
change action clipping
change temporal ensembling
```

Any change requires a new decision.

---

# 22. Required Result Package

Recommended path:

```text
experiments/force/r5_pre_final_dataset_real_robot_gate/index1/result/
```

Required:

```text
summary.md
validation.md
model_identity.json
trial_manifest.csv
per_trial_stage_metrics.csv
positioning_summary.csv
autonomous_success_summary.csv
assisted_recovery_summary.csv
local_regulator_metrics.csv
video_index.md
runtime_log_index.md
safety_notes.md
builder_review.md
red_team_review.md
proposal.md
```

---

# 23. Completion Criteria

`R5_COMPLETE = true` requires:

1. all five model identities frozen before robot testing;
2. A/B/C positions fixed;
3. 15 trials attempted, or 12 core trials plus explicit reason bad-seed control could not run;
4. P0/P1 recorded for every valid autonomous trial;
5. autonomous vs assisted separated;
6. New-GL residual logs retained;
7. one Full60 outcome assigned;
8. one seed-transfer outcome assigned;
9. no post-hoc model changes.

---

# 24. Plain-Language Summary

Do not collect the final 120–150 episode dataset yet.

First test whether the current evidence actually transfers to the robot.

The minimum useful robot study is:

```text
Strong-G-36 seed3
Strong-G-60 seed3

New-GL-36
New-GL-60

plus one clearly weak G-Load-60 seed
```

All at the same:

```text
A / B / C
```

positions.

The two main questions are:

```text
Does 60 training episodes position better than 36?

Does a bad global seed also behave worse on the robot?
```

If the answer to the first question is yes, then collecting a denser final 120–150 episode dataset is justified.

If the answer is no, investigate spatial representation before spending time collecting much more data.
