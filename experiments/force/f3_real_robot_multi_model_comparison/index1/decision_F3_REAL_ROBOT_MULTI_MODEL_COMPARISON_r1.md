# decision_F3_REAL_ROBOT_MULTI_MODEL_COMPARISON_r1.md

## 0. Contract Header

- Experiment ID: `F3_REAL_ROBOT_MULTI_MODEL_COMPARISON`
- Revision: `r1`
- Mode: `REAL_ROBOT_COMPARISON / FEASIBILITY`
- Track: `force`
- Status at creation: `FROZEN / NOT_RUN`
- Scientific scope: `EXPLORATORY_MULTI_MODEL_REAL_ROBOT_COMPARISON`
- Supersedes for execution:
  - `F3_REAL_ROBOT_LOCAL_REGULATOR_FEASIBILITY r0`
- r0 remains archived and must not be deleted.
- Parent evidence:
  - `F2_1 r1`: `SINGLE_FRAME_LOAD_SUFFICIENT`
  - `F2_2`: `GATING_HURTS`
  - `F2_3`: `LOCAL_LOAD_REGULATION_PROMISING`

Recommended repository path:

```text
experiments/force/f3_real_robot_multi_model_comparison/index1/decision.md
```

Recommended archive path for prior F3-r0:

```text
experiments/force/f3_real_robot_multi_model_comparison/index1/decision_r0.md
```

---

## 1. Primary Goal

This round minimizes repeated robot startup/testing sessions by evaluating the key surviving model families in one controlled real-robot session.

It answers:

> Among the historically relevant and currently supported model families, which actuator-feedback strategy produces the most promising real closed-loop grasp behavior under the same task conditions?

This is still an exploratory real-robot comparison.

It is not the sealed Final Test.

---

## 2. Frozen Model Set

Exactly four models are authorized.

### M0 — Historical P1 / 8D OC-ACT

Historical project reference.

Inputs:

```text
RGB
+
8D robot state
=
5 arm joints
+ gripper.pos
+ gripper.load
+ gripper.current/curr
```

Purpose:

> Represent the historically used P1 / OC-ACT configuration.

This is not a pure ACT baseline.

---

### M1 — Clean 6D No-Actuator-Feedback OC-ACT

Inputs:

```text
RGB
+
6D proprioception
=
5 arm joints
+ gripper.pos
```

No:

```text
load
current
```

Purpose:

> Real-robot reference for behavior without actuator feedback.

---

### M2 — Global Single-Frame Load OC-ACT

Inputs:

```text
RGB
+
6D proprioception
+
single-frame gripper.load
```

`load` is fused globally into OC-ACT.

No:

```text
current
W15 history
contact gate
local residual policy
```

Purpose:

> Test the best supported global actuator-feedback fusion strategy.

---

### M3 — Global 6D OC-ACT + Local Load Residual Policy

Global:

```text
RGB
+
6D proprioception
↓
frozen global OC-ACT
```

Local:

```text
load_t
+
gripper.pos_t
+
nominal_gripper_t
↓
local MLP
↓
bounded delta_gripper
```

Final:

```text
final gripper
=
nominal gripper
+
bounded delta_gripper
```

Residual bound:

```text
±5% demonstrated gripper action range
```

Purpose:

> Test the strongest F2.3 local load-regulation candidate.

---

## 3. Explicitly Excluded Models

Do not real-robot test the following in this round:

```text
A2 / W15 temporal load
B1 / deterministic position-based gate
C1-no-load
load+current local variants
FiLM variants
learned gate variants
high-frequency local controller variants
```

Reason:

> These are either already disfavored offline, mechanism-only controls, or not yet scientifically authorized.

---

## 4. Key Scientific Contrasts

### M0 vs M1

Question:

> Does the historical 8D P1 configuration provide a practical real-robot advantage over the clean 6D no-actuator-feedback model?

Interpret cautiously because M0 differs by both load and current and historical training semantics.

---

### M2 vs M1

Question:

> Does globally fused single-frame load improve real-robot behavior relative to no actuator feedback?

---

### M3 vs M1

Question:

> Does local load-aware residual regulation improve real-robot behavior relative to the clean 6D global baseline?

---

### M3 vs M2

Primary architecture contrast:

> Is local load-aware regulation more promising than global load fusion in real closed-loop execution?

---

## 5. Model Identity Freeze

Before robot testing, record for M0/M1/M2/M3:

```text
checkpoint path
checkpoint hash / identifier
git HEAD
model config
input state fields/order
camera inputs
action dimension
normalization
temporal ensembling
action clipping / smoothing
control frequency
```

For M3 also record:

```text
local checkpoint
local normalization
delta_max
local MLP architecture
```

If any model identity is ambiguous:

```text
BLOCKED_MODEL_IDENTITY
```

Do not guess or substitute.

---

## 6. Common Deployment Conditions

All four models must use the same:

```text
robot hardware
camera setup
cup type
cup fill state
task definition
control frequency
deployment machine
action post-processing
temporal ensembling settings
action clipping
```

Only model-specific architecture differences are allowed.

Do not change P1 task semantics between variants.

---

## 7. Task Scope

Use the familiar existing cup task.

Do not expand:

```text
spatial range
cup type
fill level
task objective
transport distance
```

inside F3-r1.

This round is intended to compare models under a known real-robot setting, not test generalization.

---

## 8. M3 Safety Gate — Mandatory Shadow Mode

Only M3 introduces a separate online action correction module.

Therefore M3 must pass shadow mode before its formal intervention trials.

Minimum:

```text
M3 shadow × 3 valid trials
```

During shadow:

```text
local MLP runs
delta is logged
final correction is NOT executed
```

Executed behavior remains the frozen M3 global-only 6D policy.

---

## 9. M3 Shadow PASS Criteria

Shadow passes only if:

1. arm actions are untouched;
2. all local residuals obey the frozen bound;
3. no NaN/inf;
4. no dangerous alternating-sign oscillation;
5. near-bound outputs are not dominant;
6. residual behavior is not obviously concentrated in release/opening in a harmful way;
7. local regulator disable-path works;
8. logs/video are complete.

If shadow fails:

```text
M3 formal intervention is blocked
```

M0/M1/M2 may still be evaluated if safe and scientifically useful.

---

## 10. Formal Real-Robot Budget

Each model:

```text
3 valid formal trials
```

Total planned formal trials:

```text
M0 × 3
M1 × 3
M2 × 3
M3 × 3
=
12 trials
```

Plus:

```text
M3 shadow × 3
```

Total planned robot executions:

```text
15
```

Do not automatically increase trial count.

---

## 11. Trial Order

Use blocked rotation to reduce time/order drift.

Frozen preferred order:

```text
Round 1:
M0 → M1 → M2 → M3

Round 2:
M0 → M1 → M2 → M3

Round 3:
M0 → M1 → M2 → M3
```

If M3 is blocked by shadow failure:

```text
Round 1:
M0 → M1 → M2
Round 2:
M0 → M1 → M2
Round 3:
M0 → M1 → M2
```

Do not reorder models after observing favorable/unfavorable outcomes.

---

## 12. Reset Protocol Between Trials

Between trials:

- return robot to the same starting configuration;
- restore cup to the same nominal placement protocol;
- verify camera view;
- verify cup fill state unchanged;
- verify no visible hardware anomaly;
- reset model state/history according to its normal deployment path.

Document any reset deviation.

---

## 13. Trial Validity

A formal trial is valid only if:

1. checkpoint/config is correct;
2. logging is complete;
3. camera recording is complete;
4. cup/setup matches protocol;
5. no unrelated hardware fault occurs;
6. no operator intervention occurs before a safety issue or after task completion.

Safety-intervention trials stay in the record.

Do not hide failures.

---

## 14. Common Behavioral Metrics

For every trial report:

```text
grasp established
lift achieved
approximate lift height if reliably measurable
hold duration
visible slip
premature release
excessive cup deformation
stable transport if applicable
stable placement
correct release
full task-chain completion
operator emergency intervention
```

Also retain:

```text
gripper command
gripper.pos
gripper.load
gripper.current if available
arm state/action
video
```

---

## 15. M3-Specific Metrics

For M3 formal trials report additionally:

```text
delta_gripper over time
mean |delta_gripper|
median |delta_gripper|
max |delta_gripper|
near-bound fraction
sign-change rate
nonzero fraction
major correction events
```

Arm outputs must remain unchanged by the local regulator.

---

## 16. Hold Behavior

Because prior P1 behavior showed low/short lift-hold behavior, explicitly record:

```text
hold start
hold end
approximate hold duration
whether cup remains stably gripped
whether slip occurs
whether gripper opens prematurely
```

Do not attribute arm hold duration improvements to the load regulator unless the mechanism and data support that interpretation.

---

## 17. Safety Stop Conditions

Immediately stop the current model trial if:

- dangerous over-closing;
- visible cup crushing;
- high-frequency gripper oscillation;
- unexpected repeated tightening;
- premature release caused by intervention;
- non-finite action;
- residual bound violation for M3;
- local code modifies arm action;
- hardware safety risk.

For M3:

```text
one regulator-caused serious unsafe event
→ stop all further M3 intervention trials
```

Do not tune and retry inside r1.

---

## 18. Real-Robot Comparison Philosophy

This is a:

```text
3 trials/model exploratory comparison
```

Do not report inferential statistical significance.

Allowed conclusions:

```text
pilot advantage
consistent behavioral signal
feasibility difference
descriptive comparison
```

Forbidden:

```text
statistically proven superiority
final success rate
generalization proof
```

---

## 19. Primary Evaluation Focus

Primary qualitative/quantitative focus:

```text
M3 vs M2
```

because this tests:

> local load regulation vs global load fusion.

Secondary:

```text
M2 vs M1
M3 vs M1
M0 as historical reference
```

Do not make M0 vs M1 the central scientific claim because M0 differs in multiple historical dimensions.

---

## 20. Descriptive Scoring Table

Produce one unified table with at least:

| Model | Feedback strategy | Grasp | Lift | Hold | Slip | Deformation | Place | Release | Full chain | Safety |
|---|---|---|---|---|---|---|---|---|---|---|

Per-trial rows and model summaries are both required.

---

## 21. Outcome Categories

### `M3_REAL_ROBOT_PROMISING`

Use only if:

1. M3 shadow passes;
2. no M3 regulator-caused unsafe event;
3. at least 2/3 M3 formal trials complete the grasp-lift-place chain;
4. M3 shows a repeatable non-worse/positive grasp-related signal relative to M2;
5. M3 residual behavior is bounded and mechanistically plausible.

Interpretation:

> Local load-aware regulation is promising enough for a larger real-robot study.

---

### `M2_GLOBAL_LOAD_REMAINS_BEST`

Use if:

- M2 is descriptively better or more stable than M3;
- M3 does not provide a clear local-regulation advantage.

Interpretation:

> Global single-frame load fusion remains the preferred real-robot candidate.

---

### `ACTUATOR_FEEDBACK_NOT_CLEAR_ON_ROBOT`

Use if:

- M1 performs similarly to or better than M2/M3;
- no clear load-aware advantage appears.

Interpretation:

> Offline actuator-feedback gains did not clearly translate to the small real-robot pilot.

---

### `M3_UNSAFE_OR_DESTABILIZING`

Use if:

- M3 causes regulator-related unsafe behavior.

Interpretation:

> Do not scale local residual regulation.

---

### `UNRESOLVED`

Use if:

- too many invalid trials;
- model/checkpoint mismatch;
- setup differs materially;
- logging incomplete;
- unrelated hardware problems dominate.

---

## 22. Final12 Boundary

Final12 remains sealed.

This round uses a development real-robot test session.

Do not:

```text
run Final12
tune using Final12
report Final12 success
```

---

## 23. No Post-Hoc Rescue

After formal testing begins, do not:

- change checkpoints;
- change temporal ensembling;
- change action clipping;
- increase M3 delta_max;
- retrain M3;
- add current to M2/M3;
- add W15;
- add gate;
- modify task difficulty;
- change trial count to rescue a model;
- remove failed trials;
- reorder remaining variants based on results.

Any meaningful change requires a new experiment.

---

## 24. Completion Criteria

`F3_R1_COMPLETE = true` requires:

1. all model identities frozen;
2. common deployment configuration audited;
3. M3 shadow ×3 completed or M3 formally blocked;
4. planned valid trials completed for all non-blocked models;
5. all safety events retained;
6. per-trial behavior metrics complete;
7. unified comparison table complete;
8. M3 mechanism metrics complete if M3 tested;
9. videos/logs indexed;
10. Final12 untouched;
11. Builder Review evidence complete;
12. Red-Team Review evidence complete;
13. outcome category assigned.

Completion does not imply any model is definitively superior.

---

## 25. Expected Result Package

Recommended path:

```text
experiments/force/f3_real_robot_multi_model_comparison/index1/result/
```

Required:

```text
summary.md
validation.md
model_identity/
deployment_config.json
trial_manifest.csv
per_trial_metrics.csv
model_summary.csv
m3_shadow_metrics.csv
m3_shadow_review.md
m3_regulator_metrics.csv
safety_review.md
mechanism_review.md
video_index.md
realtime_log_index.md
leakage_audit.md
figures/
proposal.md
```

Recommended figures:

```text
figures/model_behavior_summary.png
figures/hold_duration_by_model.png
figures/grasp_chain_by_model.png
figures/load_gripper_behavior_by_model.png
figures/m3_delta_timeline.png
figures/m3_delta_distribution.png
```

---

## 26. Builder Review Questions

Builder Review must verify:

1. Were exactly M0/M1/M2/M3 used?
2. Were identities/configs unambiguous?
3. Were all models run under the same task/setup?
4. Did M3 pass shadow before intervention?
5. Was M3 arm preservation maintained?
6. Were trial order/reset rules followed?
7. Were failed/aborted trials retained?
8. Was any model tuned after formal testing began?
9. Was Final12 untouched?
10. Are logs/videos sufficient to audit behavior?

---

## 27. Red-Team Review Questions

Red-Team Review must ask:

1. Are differences caused by model strategy or setup/order drift?
2. Does M3 genuinely improve grasp behavior over M2?
3. Is M3 benefit just a static gripper bias?
4. Does M2 global load fusion remain more robust overall?
5. Does M1 unexpectedly match or outperform load-aware models?
6. Does M0 historical P1 differ for reasons beyond actuator feedback?
7. Are hold/lift differences actually caused by arm behavior rather than gripper regulation?
8. Are three trials/model enough only for feasibility, not effectiveness?

---

## 28. Plain-Language Summary

Instead of starting the robot many separate times, this round tests all key surviving models in one session.

The four models are:

```text
M0:
historical 8D P1
(load + current already inside state)

M1:
clean 6D OC-ACT
(no actuator feedback)

M2:
6D OC-ACT + single-frame load globally fused

M3:
6D global OC-ACT
+
local load-aware residual gripper policy
```

Each model gets 3 formal trials.

M3 first gets 3 shadow trials for safety.

The purpose is to answer, in one robot session:

```text
Does actuator feedback help?
Does global load fusion help?
Does local load regulation help more?
```

This is still a small exploratory comparison, not the final success-rate experiment.

---

## 29. Handoff Rule

After F3-r1:

1. stop;
2. return the full result package;
3. perform Builder Review;
4. perform Red-Team Review;
5. discuss with the user;
6. do not automatically expand trial count;
7. do not retrain/tune any model;
8. do not read Final12.

Only after review may a later decision freeze the final real-robot model and a larger formal evaluation.
