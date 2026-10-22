# decision_F5_UNIFIED_REAL_ROBOT_VALIDATION.md

## 0. Contract Header

- Experiment ID: `F5_UNIFIED_REAL_ROBOT_VALIDATION`
- Revision: `r0`
- Mode: `REAL_ROBOT_MULTI_MODEL_VALIDATION`
- Track: `force / final candidate validation`
- Status: `FROZEN / NOT_RUN`
- New training authorized: `NO`
- New architecture authorized: `NO`
- Final12: `SEALED`
- Purpose:
  - compare the key real-robot candidates in one session
  - minimize repeated robot restarts
  - validate whether offline improvements transfer to closed-loop behavior

Recommended repository path:

```text
experiments/force/f5_unified_real_robot_validation/index1/decision.md
```

---

## 1. Primary Research Question

This round answers:

> Does New-GL transfer its offline advantages to real closed-loop behavior, and does it outperform global-only and local-only load usage while at least matching the historical practical reference?

The main scientific comparisons are:

```text
Strong-G vs L-Load
New-GL vs Strong-G
New-GL vs L-Load
```

`P1-Hist` is a historical practical reference, not a strict causal ablation.

---

## 2. Frozen Model Set

Exactly four models are authorized.

### `P1-Hist`

Historical P1 / M0.

```text
Global input:
RGB
+ 5 arm joints
+ gripper.pos
+ gripper.load
+ gripper.current
```

No local residual branch.

Role:
historical practical reference.

---

### `Strong-G`

Best verified G-Load.

```text
Global:
RGB
+ 6D proprioception
+ single-frame load
↓
OC-ACT
↓
arm + gripper
```

Use:

```text
G-Load seed3 / 80k
```

No local residual.

Role:
strongest global-load-only candidate.

---

### `L-Load`

Local-load-only candidate.

```text
Global:
RGB + 6D proprioception
(no load)

Local:
load
+ gripper.pos
+ nominal gripper
↓
small MLP
↓
bounded delta_gripper
```

Use the best verified L-Load checkpoint from F4.1-r1.

Role:
local-regulation-only candidate.

---

### `New-GL`

Preferred dual-role candidate from F4.2.

```text
Global:
Strong-G seed3 / 80k
with single-frame load

Local:
load
+ gripper.pos
+ nominal gripper
↓
small MLP
↓
bounded delta_gripper
```

Use the F4.2 selected best checkpoint:

```text
global = G-Load seed3 / 80k
local = selected best New-GL local checkpoint
```

Role:
primary candidate.

---

## 3. Explicitly Excluded Models

Do not test in F5:

```text
B0-6D
Old-GL
old G-Load seed1
W15 variants
gated variants
current-only variants
FiLM variants
learned-gate variants
```

Reason:
they are no longer the highest-value candidates for this real-robot comparison.

---

## 4. Fixed Real-Robot Positions

Use exactly three predefined positions:

```text
Position A
Position B
Position C
```

Requirements:
- all within the currently familiar validated workspace;
- no expansion of task range;
- same cup type;
- same fill state;
- same table/camera setup;
- same nominal task.

All four models must be evaluated at all three positions.

Do not move a position after seeing one model fail.

---

## 5. Formal Trial Budget

Each model:

```text
3 formal trials
```

One at each position.

Total:

```text
P1-Hist × 3
Strong-G × 3
L-Load × 3
New-GL × 3

= 12 formal trials
```

Do not automatically increase trial count.

---

## 6. Trial Order

Use blocked rotation.

### Round 1 — Position A

```text
P1-Hist
→ Strong-G
→ L-Load
→ New-GL
```

### Round 2 — Position B

```text
Strong-G
→ L-Load
→ New-GL
→ P1-Hist
```

### Round 3 — Position C

```text
L-Load
→ New-GL
→ P1-Hist
→ Strong-G
```

Purpose:
reduce fixed order/time drift.

Do not reorder after observing results.

---

## 7. Minimal Pre-Run Sanity Check

Before formal testing, verify only:

```text
checkpoint/config identity
input state field order
camera input path
action dimension
gripper direction/sign
local residual hard bound for L-Load/New-GL
```

No mandatory multi-trial shadow stage.

No repeated safety unlock.

Once the model passes the basic sanity check, proceed directly to formal testing.

---

## 8. Minimal Safety Constraints

Retain only:

```text
correct model/config identity
hard action/residual limits
human emergency stop
stop on obvious dangerous behavior
```

Do not introduce new multi-layer safety gates.

For L-Load/New-GL, local residual remains bounded by the frozen offline limit.

---

## 9. Trial Phases

Each formal trial has up to two phases.

### Phase A — Autonomous

From trial start until:
- autonomous task completion; or
- the model is clearly no longer capable of recovering autonomously.

This phase determines autonomous success/failure.

### Phase B — Assisted Recovery

Optional.

Starts only after autonomous failure is confirmed.

Human assistance may:
- reposition the cup;
- move the cup into a graspable region;
- provide limited physical support;
- otherwise place the system into a recoverable state.

Purpose:

> test whether downstream close/lift/hold/place/release behavior still exists.

Assisted recovery must never be counted as autonomous success.

---

## 10. Stage-Based Behavioral Evaluation

For every formal trial record:

```text
S0 Approach
S1 Grasp Geometry
S2 Close
S3 Lift
S4 Hold
S5 Place
S6 Release
```

Use:

```text
PASS
FAIL
UNKNOWN
```

Also record:

```text
full_autonomous_success
assisted_recovery_used
full_chain_after_assistance
```

---

## 11. Stage Definitions

### S0 — Approach

Robot reaches the intended cup vicinity.

### S1 — Grasp Geometry

Cup reaches a plausible two-finger grasp region.

### S2 — Close

Gripper closes sufficiently to attempt grasp.

### S3 — Lift

Cup is physically lifted with the gripper/arm.

### S4 — Hold

Cup remains stably grasped through the expected hold segment.

### S5 — Place

Cup is returned/moved to the intended placement area.

### S6 — Release

Gripper releases appropriately after placement.

---

## 12. Common Behavioral Metrics

For every trial record:

```text
approach_ok
grasp_geometry_ok
close_ok
lift_ok
hold_ok
place_ok
release_ok
full_autonomous_success

visible_slip
premature_release
visible_over_closure
cup_deformation
operator_emergency_stop
assisted_recovery_used
full_chain_after_assistance
```

If reliably measurable:

```text
approximate lift height
approximate hold duration
```

---

## 13. Video Evidence

Video is used for:

```text
behavior-stage analysis
slip
hold
placement
release
assisted-recovery interpretation
```

Video does not need frame-level synchronization with CSV.

The only required identity link is:

```text
MODEL_ID
TRIAL_ID
POSITION
```

---

## 14. CSV / Runtime Evidence

Runtime logs are analyzed separately from video.

For all models retain where available:

```text
gripper command
gripper.pos
load
current
arm action/state
```

For L-Load/New-GL additionally:

```text
nominal_gripper
delta_gripper
final_gripper
```

Do not require sub-second video↔CSV alignment.

Use model/trial-level correspondence only.

---

## 15. Minimal Naming Convention

Every trial must have one unique ID.

Recommended format:

```text
F5_<MODEL>_<POSITION>_<TRIAL>
```

Examples:

```text
F5_P1Hist_A_01
F5_StrongG_A_01
F5_LLoad_A_01
F5_NewGL_A_01
```

Use the same ID in:
- video filename;
- CSV filename;
- stdout/log folder;
- trial manifest.

This is metadata only and must not add operational friction.

---

## 16. Local-Regulator Metrics

For L-Load and New-GL record:

```text
mean |delta_gripper|
median |delta_gripper|
max |delta_gripper|
near-bound fraction
sign-change rate
nonzero fraction
```

Also note:
- whether correction is mostly small;
- whether correction saturates;
- whether correction appears unstable.

Do not infer physical force from load.

---

## 17. Primary Scientific Comparison

The most important comparison is:

```text
New-GL vs Strong-G
```

Question:

> Does adding local load-aware correction on top of the strongest global-load policy improve real behavior?

Second:

```text
New-GL vs L-Load
```

Question:

> Does keeping global load context improve behavior over local-only load usage?

Third:

```text
Strong-G vs L-Load
```

Question:

> Which single-role load strategy transfers better to real execution?

---

## 18. P1-Hist Interpretation

P1-Hist is not a strict ablation.

Allowed interpretation:

```text
historical practical reference
```

Use it to answer:

> Has the new pipeline at least recovered or exceeded the historically competent action-chain behavior?

Do not attribute P1-Hist differences specifically to:
- current;
- 8D input;
- load;
without further controlled evidence.

---

## 19. Autonomous Success Metric

For each model report:

```text
autonomous_success_count / 3
```

Because n=3:
- descriptive only;
- no inferential statistical significance.

Example allowed wording:

```text
New-GL completed 3/3 pilot trials autonomously.
```

Forbidden:

```text
New-GL is statistically significantly superior.
```

---

## 20. Stage Success Summary

For each model summarize:

```text
Approach: x/3
Geometry: x/3
Close: x/3
Lift: x/3
Hold: x/3
Place: x/3
Release: x/3
Full autonomous: x/3
```

This is the primary diagnostic summary.

---

## 21. Assisted-Recovery Summary

For every assisted trial record:

```text
first autonomous failure stage
assistance type
stages recovered after assistance
full downstream chain recovered?
```

Example:

```text
Autonomous:
Approach PASS
Geometry FAIL

Assistance:
cup moved into gripper region

After assistance:
Close PASS
Lift PASS
Hold PASS
Place PASS
Release PASS
```

Interpretation:

```text
upstream geometry failure
downstream action chain preserved
```

---

## 22. Outcome Categories

### `NEW_GL_REAL_ROBOT_PREFERRED`

Use if:
1. New-GL completes at least 2/3 autonomous trials;
2. New-GL is descriptively equal or better than Strong-G on full-chain completion;
3. New-GL is descriptively equal or better than L-Load;
4. no regulator-caused unsafe/unstable behavior;
5. local residual remains bounded/non-degenerate;
6. stage-level behavior supports the result.

Interpretation:
> New-GL becomes the preferred real-robot main model.

### `STRONG_G_REAL_ROBOT_PREFERRED`

Use if:
- Strong-G is more stable/complete than New-GL;
- local residual does not add practical benefit.

### `L_LOAD_REAL_ROBOT_PREFERRED`

Use if:
- L-Load outperforms both Strong-G and New-GL.

### `P1_HIST_STILL_PRACTICALLY_STRONGEST`

Use if:
- P1-Hist clearly retains the most complete practical behavior;
- new candidates fail to match its action-chain quality.

This does not identify the causal reason.

### `MIXED_OR_UNRESOLVED`

Use if:
- outcomes are strongly position-dependent;
- hardware/setup problems dominate;
- too many trials are invalid;
- no candidate clearly separates.

---

## 23. No Post-Hoc Rescue

After formal testing starts, do not:

```text
change checkpoint
change action clipping
change temporal ensembling
increase local residual bound
change task positions
change cup/fill state
change model order
retrain
add extra trials to rescue a negative model
remove failed trials
```

Any meaningful change requires a new experiment.

---

## 24. Completion Criteria

`F5_COMPLETE = true` requires:

1. all four model identities verified;
2. three positions frozen;
3. all 12 formal trials attempted;
4. stage-based metrics complete;
5. autonomous vs assisted phases separated;
6. local-regulator metrics complete for L-Load/New-GL;
7. all failed/aborted trials retained;
8. unified model summary complete;
9. one outcome category assigned;
10. Final12 untouched.

Completion does not imply final publication-level statistical superiority.

---

## 25. Expected Result Package

Recommended path:

```text
experiments/force/f5_unified_real_robot_validation/index1/result/
```

Required:

```text
summary.md
validation.md
model_identity.json
trial_manifest.csv
per_trial_stage_metrics.csv
model_stage_summary.csv
autonomous_success_summary.csv
assisted_recovery_summary.csv
local_regulator_metrics.csv
safety_notes.md
video_index.md
runtime_log_index.md
builder_review.md
red_team_review.md
proposal.md
figures/
```

Recommended figures:

```text
figures/model_stage_success.png
figures/full_chain_success.png
figures/assisted_recovery_summary.png
figures/local_residual_activity.png
figures/newgl_vs_strongg_behavior.png
```

---

## 26. Builder Review Questions

Builder Review must verify:

1. Were exactly the four frozen models used?
2. Were the correct checkpoints loaded?
3. Were all models tested at A/B/C?
4. Was blocked rotation followed?
5. Were autonomous and assisted phases separated?
6. Were failed trials retained?
7. Were local residual bounds preserved?
8. Was no post-hoc tuning performed?
9. Was Final12 untouched?
10. Are all videos/logs identifiable by trial ID?

---

## 27. Red-Team Review Questions

Red-Team Review must ask:

1. Does New-GL actually improve closed-loop behavior over Strong-G?
2. Is New-GL advantage driven by one position only?
3. Does local residual help grasp stability or merely alter gripper bias?
4. Does Strong-G retain a better global action chain?
5. Does L-Load expose any loss of global phase/context?
6. Does P1-Hist still outperform the new pipeline?
7. Are assisted-recovery outcomes consistent with autonomous failure-stage labels?
8. Is 3 trials/model enough only for pilot model selection, not final statistical claims?

---

## 28. Plain-Language Summary

This round tests four models once, in one robot session:

```text
P1-Hist:
historical practical reference

Strong-G:
global load only

L-Load:
local load only

New-GL:
global load + local load residual
```

Each model runs at:

```text
Position A
Position B
Position C
```

for 3 trials/model, 12 trials total.

There is no mandatory multi-layer shadow process.

If a model clearly cannot recover autonomously, human assistance may be used afterward to test whether the downstream action chain is still present.

The main question is:

> Does New-GL combine Strong-G's global action quality with Local MLP's gripper correction in real closed-loop execution?

If yes, New-GL becomes the preferred real-robot main model.
