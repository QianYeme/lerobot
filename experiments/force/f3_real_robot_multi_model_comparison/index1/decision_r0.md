# decision_F3_REAL_ROBOT_LOCAL_REGULATOR_FEASIBILITY.md

## 0. Contract Header

- Experiment ID: `F3_REAL_ROBOT_LOCAL_REGULATOR_FEASIBILITY`
- Revision: `r0`
- Mode: `REAL_ROBOT_FEASIBILITY`
- Track: `force`
- Status at creation: `FROZEN / NOT_RUN`
- Scientific scope: `EXPLORATORY_REAL_ROBOT_FEASIBILITY`
- Parent evidence:
  - `F2_1_LOAD_TEMPORAL_ENCODING_ABLATION r1`
    - outcome: `SINGLE_FRAME_LOAD_SUFFICIENT`
  - `F2_2_CONTACT_GATED_SINGLE_FRAME_LOAD`
    - outcome: `GATING_HURTS`
  - `F2_3_GLOBAL_VS_LOCAL_LOAD_REGULATION`
    - outcome: `LOCAL_LOAD_REGULATION_PROMISING`
- Current candidate:
  - frozen 6D global OC-ACT
  - local residual gripper MLP
  - local inputs: single-frame `load`, `gripper.pos`, nominal gripper action
  - bounded residual: `±5%` of demonstrated gripper action range

Recommended repository path:

```text
experiments/force/f3_real_robot_local_regulator_feasibility/index1/decision.md
```

This is a new experiment contract.
It does not overwrite F2.3.

---

## 1. Single Research Question

This round answers only:

> Can the F2.3 C1-load local residual gripper regulator run safely and stably in real closed-loop execution without breaking the frozen global policy’s task behavior, and does it show any plausible grasp/hold benefit worth larger evaluation?

This round is **not** a final success-rate experiment.

It does not claim:
- statistically significant task improvement;
- final generalization improvement;
- calibrated force control;
- Final Test performance.

---

## 2. Frozen Candidate Architecture

Use the F2.3 C1-load semantics exactly.

### Global policy

```text
RGB
+
6D proprioception
↓
frozen global OC-ACT
↓
arm actions + nominal gripper action
```

The global policy receives:

```text
no load
no current
```

### Local regulator

Inputs:

```text
single-frame load_t
gripper.pos_t
nominal_gripper_t
```

Model:

```text
small MLP
```

Output:

```text
delta_gripper_t
```

Final gripper command:

```text
final_gripper_t
=
nominal_gripper_t
+
bounded delta_gripper_t
```

The local module must not alter arm actions.

---

## 3. Global Checkpoint Identity

Use the same verified 6D global checkpoint used by F2.3 C1-load.

Record:

```text
checkpoint path / hash
git commit / HEAD
training config
state semantics
action semantics
camera inputs
normalization
temporal ensembling settings
action clipping / smoothing
control frequency
```

If the exact checkpoint cannot be verified:

```text
BLOCKED_GLOBAL_CHECKPOINT_IDENTITY
```

Do not substitute another P1 checkpoint.

---

## 4. Local Checkpoint Identity

Use the exact accepted C1-load local MLP checkpoint from F2.3.

Record:

```text
local checkpoint path / hash
input normalization
residual bound
MLP architecture
training seed
F2.3 outcome reference
```

If the accepted local checkpoint cannot be identified:

```text
BLOCKED_LOCAL_CHECKPOINT_IDENTITY
```

---

## 5. Residual Bound

Keep the F2.3 bound:

```text
|delta_gripper|
<=
5% of demonstrated gripper action range
```

No increase is allowed.

For safety, the real-robot deployment may use a smaller bound only if frozen **before** the first intervention trial and documented as:

```text
F3 safety-reduced bound
```

If a smaller bound is chosen, it must remain fixed for all regulated trials.

Do not enlarge it after seeing trial behavior.

---

## 6. Arm Preservation

The local regulator may modify only the gripper component.

Required invariant:

```text
regulated arm action
==
nominal global arm action
```

up to serialization precision.

Any material arm difference causes:

```text
INVALID_LOCAL_ISOLATION
```

and real-robot intervention must stop.

---

## 7. Stage 0 — Deployment Equivalence / Dry Validation

Before shadow mode, verify:

1. both checkpoints load correctly on deployment hardware;
2. input field ordering is correct;
3. normalization matches training;
4. local output uses the correct gripper sign / convention;
5. residual clipping is active;
6. regulator can be disabled with one config flag;
7. disabled mode restores the exact global-only path;
8. arm outputs remain unchanged;
9. no Final12 content is read.

Required artifact:

```text
result/deployment_validation.md
```

If any of these fail:

```text
STOP
```

No real-robot rollout.

---

## 8. Stage 1 — Real-Robot Shadow Mode

Run the frozen global policy on the real robot in the current familiar cup/task setting.

The local regulator is active computationally but:

```text
does NOT modify the executed gripper command
```

For every control step log:

```text
timestamp
global arm action
nominal gripper action
load
gripper.pos
local predicted delta
clipped delta
hypothetical final gripper action
```

Also retain video.

Shadow mode purpose:

> Verify that the local regulator behaves sensibly under real closed-loop state distributions before it is allowed to act.

---

## 9. Shadow-Mode Budget

Minimum:

```text
3 valid shadow trials
```

These are not baseline effectiveness trials.

They are safety/mechanism checks.

Use the same familiar cup type, fill state, and task setup currently used for P1 validation.

Do not expand spatial range or change the task.

---

## 10. Shadow-Mode PASS Criteria

Shadow mode passes only if all are true:

1. no arm-action modification;
2. all residuals remain within the frozen bound;
3. no numerical instability / NaN / inf;
4. no high-frequency alternating-sign oscillation that would be dangerous if applied;
5. near-bound behavior is not dominant;
6. predicted corrections are not obviously concentrated in release/opening segments in a way that conflicts with the task;
7. the regulator can be disabled cleanly;
8. logging/video are complete.

Suggested diagnostic flags:

```text
SHADOW_OSCILLATORY
SHADOW_BOUND_SATURATION
SHADOW_RELEASE_CONFLICT
SHADOW_NUMERICAL_FAILURE
```

If any serious flag is triggered:

```text
STOP
```

Do not enter intervention trials.

---

## 11. Shadow Diagnostics

Report per trial:

```text
mean |delta_gripper|
median |delta_gripper|
max |delta_gripper|
near-bound fraction
sign-change rate
activation / nonzero fraction
delta during approach
delta during grasp/hold
delta during release
```

If phase segmentation is not reliably available online, use video review afterward for descriptive interpretation only.

Do not introduce manual phase labels into the controller.

---

## 12. Stage 2 — Baseline Real-Robot Trials

If shadow mode passes, run:

```text
Baseline global-only: 3 valid trials
```

Baseline means:

```text
local regulator disabled
```

Use the same frozen global checkpoint.

No changes to:
- temporal ensembling;
- action clipping;
- camera setup;
- robot configuration;
- cup type / fill state;
- task definition.

---

## 13. Stage 3 — Regulated Real-Robot Trials

Then run:

```text
Global + C1-load regulator: 3 valid trials
```

Use the same setup as baseline.

Only difference:

```text
local regulator enabled
```

No model/threshold/bound changes between regulated trials.

---

## 14. Trial Order

Preferred order for r0:

```text
Baseline × 3
then
Regulator × 3
```

Reason:

- keeps the first intervention exposure small and easy to monitor;
- avoids extra scientific variables.

If hardware drift is a major concern, alternating order may be proposed before execution, but once formal trials begin the order must not change.

---

## 15. Valid Trial Criteria

A trial is valid only if:

1. camera/state/action/load logging is complete;
2. correct checkpoint/config is confirmed;
3. no operator correction occurs before the relevant grasp segment;
4. no unrelated hardware fault occurs;
5. cup/task setup matches the frozen protocol.

Emergency intervention trials remain part of the safety record even if excluded from ordinary behavioral comparison.

Do not hide aborted regulated trials.

---

## 16. Human Safety / Stop Conditions

Immediately stop regulated testing if any of the following occurs:

- visible dangerous over-closing / crushing;
- rapid gripper oscillation;
- unexpected repeated tightening;
- premature opening caused by the regulator;
- regulator causes task-chain disruption;
- local output becomes non-finite;
- residual bound is violated;
- arm output changes due to local code;
- hardware or cup safety is at risk.

After a stop condition:

```text
do not tune and resume inside F3-r0
```

Record the failure and stop for review.

---

## 17. Primary Feasibility Metrics

This round does not use one scalar success rate as the sole primary result.

Per trial report:

1. grasp established;
2. lift achieved;
3. obvious slip / premature release;
4. hold behavior;
5. excessive closure / visible cup deformation;
6. stable placement;
7. correct release;
8. full task-chain completion;
9. emergency intervention;
10. regulator nonzero count;
11. mean |delta_gripper|;
12. max |delta_gripper|;
13. near-bound fraction.

---

## 18. Hold / Grasp Behavior

Because prior P1 behavior showed low/short lift-hold behavior, explicitly record:

```text
lift achieved
approximate hold duration
whether grip remains closed/stable
whether cup visibly slips
whether release occurs prematurely
```

Do not claim that the regulator fixes short demonstration hold supervision unless evidence directly supports that.

Force/load regulation and arm hold duration remain conceptually distinct.

---

## 19. Mechanism Metrics

For each regulated trial report:

```text
load at major residual events
gripper.pos at major residual events
nominal gripper command
delta_gripper
final gripper command
timing relative to visible contact/grasp where reviewable
```

The purpose is to verify:

> The regulator is producing plausible local corrections rather than merely altering actions arbitrarily.

---

## 20. Baseline vs Regulator Comparison

Compare the 3 baseline and 3 regulated trials descriptively.

Do not report inferential statistical significance from 3 vs 3.

Allowed wording:

```text
feasibility signal
behavioral indication
qualitative/quantitative pilot difference
```

Forbidden wording:

```text
proved significant improvement
final success-rate gain
generalization improvement
```

---

## 21. Main Outcome Categories

### `F3_FEASIBLE_WITH_POSITIVE_SIGNAL`

Use only if:

1. deployment validation passes;
2. shadow mode passes;
3. no regulator-caused unsafe event occurs;
4. all residuals remain bounded;
5. at least 2/3 regulated trials complete the grasp-lift-place chain;
6. regulator behavior is mechanistically plausible;
7. regulated trials show at least one repeatable non-worse / positive grasp-related signal compared with baseline, such as:
   - less unnecessary closure,
   - reduced visible slip,
   - more stable grasp support,
   - better gripper-action behavior without task-chain disruption.

Interpretation:

> The local load-aware regulator is safe and plausible enough for a larger controlled real-robot evaluation.

---

### `F3_SAFE_BUT_NO_CLEAR_BENEFIT`

Use if:

- deployment/shadow/safety pass;
- regulated trials are stable;
- but no clear grasp-related benefit is observed.

Interpretation:

> The offline gain did not yet translate into a visible real-robot advantage at this small pilot scale.

---

### `F3_UNSAFE_OR_DESTABILIZING`

Use if any regulator-caused safety/behavior failure occurs, including:

- dangerous over-closing;
- oscillation;
- premature release;
- task-chain disruption;
- bound violation.

Interpretation:

> Do not scale this regulator to a larger robot study.

---

### `F3_UNRESOLVED`

Use if:

- too few valid trials;
- logging incomplete;
- checkpoint/config mismatch;
- shadow evidence insufficient;
- hardware faults dominate;
- baseline and intervention conditions differ materially.

---

## 22. Final12 Boundary

Final12 remains sealed.

This round does not authorize:

```text
Final12 inference
Final12 tuning
Final12 threshold selection
Final12 success reporting
```

The 3+3 trials are development feasibility trials.

---

## 23. No Post-Hoc Rescue

After seeing shadow or real-robot results, do not:

- increase residual bound;
- change local MLP;
- retrain local MLP;
- switch global checkpoint;
- add current;
- add W15;
- add contact gate;
- change temporal ensembling;
- change action clipping;
- change task setup;
- expand to more trials to rescue a negative result;
- read Final12.

Any meaningful change requires a new revision / experiment.

---

## 24. Completion Criteria

`F3_COMPLETE = true` requires:

1. global checkpoint identity verified;
2. local checkpoint identity verified;
3. deployment validation complete;
4. regulator disable-path verified;
5. arm preservation verified;
6. 3 valid shadow trials completed;
7. shadow review passed or explicit stop outcome recorded;
8. if shadow passes, baseline ×3 completed unless stopped by unrelated hardware issue;
9. if shadow passes, regulator ×3 completed unless safety stop occurs;
10. all safety events retained;
11. behavior metrics complete;
12. mechanism metrics complete;
13. video index complete;
14. Final12 untouched;
15. outcome category assigned;
16. result package complete.

Completion does not imply effectiveness.

---

## 25. Expected Result Package

Recommended path:

```text
experiments/force/f3_real_robot_local_regulator_feasibility/index1/result/
```

Required:

```text
summary.md
validation.md
deployment_validation.md
global_checkpoint_identity.json
local_checkpoint_identity.json
regulator_config.json
trial_manifest.csv
shadow_metrics.csv
shadow_review.md
per_trial_metrics.csv
mechanism_events.csv
safety_review.md
arm_preservation_audit.md
realtime_log_index.md
video_index.md
leakage_audit.md
figures/
proposal.md
```

Recommended figures:

```text
figures/shadow_delta_timeline.png
figures/shadow_delta_distribution.png
figures/baseline_vs_regulated_gripper_behavior.png
figures/load_delta_gripper_timeline.png
figures/per_trial_residual_activity.png
figures/hold_behavior_summary.png
```

---

## 26. Builder Review Questions

Builder Review must verify:

1. Were the exact F2.3 C1-load global/local checkpoints used?
2. Was the local residual bound preserved?
3. Could the regulator be disabled exactly?
4. Were arm outputs unchanged?
5. Did shadow mode precede intervention?
6. Were all shadow warnings reported?
7. Were baseline and regulator conditions matched?
8. Were aborted/unsafe trials retained?
9. Was Final12 untouched?
10. Was no post-hoc tuning performed?

---

## 27. Red-Team Review Questions

Red-Team Review must ask:

1. Did offline residual accuracy actually translate to useful closed-loop behavior?
2. Is any apparent benefit simply due to a static gripper bias?
3. Did the regulator act during plausible grasp/contact moments?
4. Did it improve grip while worsening release or cup deformation?
5. Are results dominated by one regulated trial?
6. Did baseline drift or setup differences explain the comparison?
7. Did the regulator preserve the global action chain?
8. Is the correct claim feasibility rather than effectiveness?

---

## 28. Plain-Language Summary

This round finally puts the F2.3 local regulator onto the real robot, but very cautiously.

First:

```text
global policy runs normally
local regulator only watches
```

for 3 shadow trials.

It records:

```text
"I would change the gripper by this much"
```

but does not actually change anything.

Only if that looks safe do we run:

```text
3 baseline trials
+
3 regulated trials
```

The regulator is still limited to a tiny gripper correction and cannot touch the arm.

This round asks:

> Does the offline-improved local load-aware regulator remain safe, stable, and behaviorally plausible in real closed-loop execution?

It does not yet ask:

> Has the final force-aware method statistically beaten the baseline?

---

## 29. Handoff Rule

After execution:

1. stop;
2. return the full result package;
3. perform Builder Review;
4. perform Red-Team Review;
5. discuss with the user;
6. do not automatically increase trial count;
7. do not read Final12;
8. do not retrain or tune the regulator.

Only if F3 is feasible may a later decision authorize a larger real-robot baseline-vs-regulator evaluation.
