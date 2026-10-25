# decision_R5_11_REAL_ROBOT_133K_BREAKTHROUGH_DIAGNOSTIC.md

## 0. Contract Header

- Experiment ID: `R5_11_REAL_ROBOT_133K_BREAKTHROUGH_DIAGNOSTIC`
- Revision: `r0`
- Mode: `MINIMAL_REAL_ROBOT_MODEL_SHORTLIST`
- Status: `FROZEN / NOT_RUN`
- New training: `NO`
- New architecture: `NO`
- New dataset collection: `NO`
- Primary purpose:
  - test whether the existing Full60/133k checkpoint provides real closed-loop value that was not captured by offline metrics
  - compare it against the strongest practical historical references and the best current Full60/80k real-robot candidate
- Scientific boundary:
  - this is a real-robot diagnostic/selection round, not a formal population-level success-rate study

Recommended repository path:

```text
experiments/force/r5_11_real_robot_133k_breakthrough_diagnostic/index1/decision.md
```

---

# 1. Why This Round Exists

Current evidence has reached a bottleneck:

```text
offline diagnostics became increasingly detailed
BUT
offline ranking did not reliably predict real-robot ranking
```

R5.10 onsite observation:

```text
P1 and Train36 New-GL remain the strongest practical references

Full60 M1 = clip10 / seed3 / 80k:
occasionally completed grasp → lift → place,
but often only approached the cup

clip5 / SWA candidates:
did not show corresponding real-robot advantage
```

R5.3 already produced an existing:

```text
Full60 / seed3 / 133k
```

checkpoint.

No retraining is required.

The purpose of R5.11 is to directly ask:

> Does this already-existing 133k model behave meaningfully better on the real robot than the Full60/80k model?

---

# 2. Frozen Model Shortlist

Use exactly four models in Stage A.

## R0 — P1-Hist practical reference

Canonical identity:

```text
P1-Hist / historical P1 (M0)
```

Role:

```text
historical practical upper/reference model
```

This model differs architecturally from the later G-Load family.

Therefore it is a real-robot capability reference, not a clean data-size causal control.

---

## R1 — Train36 New-GL reference

Canonical identity:

```text
New-GL-36
F4.2 preferred Strong-Global GL model
```

Role:

```text
strong Train36 real-robot reference
```

This is the most important reference for answering whether Full60 has finally recovered practical closed-loop ability.

---

## R2 — Full60 80k reference

Canonical identity:

```text
Full60 / G-Load / clip10 / seed3 / 80k
```

Use the exact checkpoint previously deployed as:

```text
R5.10 M1
```

Role:

```text
best current Full60 single-checkpoint real-robot candidate
```

Do not substitute another seed3/80k checkpoint.

Resolve exact checkpoint path/hash from the R5.10 delivery manifest.

---

## R3 — Full60 133k diagnostic model

Canonical identity:

```text
R5.3 Full60 / seed3 / 133k exposure-matched checkpoint
```

Important:

```text
this is the exact historical R5.3 133k checkpoint
```

Do not retrain it.

Do not substitute:
- 120k;
- a new deterministic 133k run;
- another seed3 trajectory;
- clip5 133k.

Resolve exact checkpoint path/hash from the R5.3 artifacts before robot execution.

Role:

```text
diagnostic test of longer-training exposure on real closed-loop behavior
```

---

# 3. Why clip5 / SWA Models Are Not Primary Here

R5.10 already gave substantial practical evidence that:

```text
clip5 / SWA offline improvement
did not translate into obvious real-robot improvement
```

Therefore R5.11 does not repeat:

```text
clip5 seed2
clip5 SWA seed2
clip5 SWA seed3
```

unless a later decision explicitly reopens them.

The current bottleneck is more fundamental:

> Can Full60 at all recover/beat the older strong models on the real robot?

---

# 4. Stage A — Minimal Position-A Screen

Use:

```text
Position A only
```

Initial order:

```text
R0 P1-Hist
R1 Train36 New-GL
R2 Full60 80k
R3 Full60 133k
```

One valid autonomous attempt per model.

Total initial attempts:

```text
4
```

This is deliberately small.

---

# 5. Manual Reset Protocol

R5.10 onsite workflow established that the operator uses manual reset and does not require a numeric Home/tolerance gate.

Therefore R5.11 uses:

```text
manual reset to the normal established start pose
same operator protocol
same cup A placement
same gripper-open convention
same camera/table setup
```

Before each attempt record:

```text
initial joint state
gripper position
load
initial camera frame
model ID
position
```

Do not reject a trial solely because no numeric Home tolerance file exists.

However, if the operator identifies an obviously abnormal reset/setup:

```text
INVALID_SETUP
```

and retry without counting it as a policy failure.

---

# 6. Deployment Freeze

All models must use their historically correct model-specific state/action dimensions, but share the same physical deployment conditions.

Freeze:

```text
same robot
same camera devices
same cup
same fill level
same table
same lighting
same Position A
same control frequency
same action safety limits
same runtime hardware/power mode
```

For models trained with different state schemas:

```text
use each model's correct historical input schema
```

Do not force P1-Hist into the G-Load schema.

Document schema differences explicitly.

---

# 7. R4 Runtime Rule

Where supported by the relevant deployment code:

```text
torch.backends.cudnn.benchmark = False
```

Use the corrected deterministic inference path.

Do not modify model weights.

---

# 8. Behavioral Stage Labels

For each autonomous attempt label:

```text
P0 = target-directed approach
P1 = valid pre-grasp / grasp geometry
P2 = close attempt
P3 = physical grasp established
P4 = cup lifted
P5 = hold achieved
P6 = place / release completed
```

Also assign:

```text
AUTONOMOUS_SUCCESS
ASSISTED_RECOVERY
FAILURE
INVALID_SETUP
```

---

# 9. Assisted-Recovery Rule

If autonomous completion is already clearly impossible, operator intervention is allowed.

Record:

```text
ASSISTED_RECOVERY = true
AUTONOMOUS_SUCCESS = false
```

The assisted continuation may still be used to determine whether later stages are executable.

Do not discard the trial.

---

# 10. Stage A Decision Gate

After the first four valid A attempts:

## Case A — R3 133k is clearly worse than R2 80k

Examples:
- fails P0 while R2 reaches P1/P2;
- systematic premature retreat;
- substantially worse approach geometry.

Outcome:

```text
FULL60_133K_ROBOT_NO_GAIN
```

Action:

```text
stop 133k testing
do not run B/C
```

Interpretation:

> longer training is not the practical solution to the Full60 regression.

---

## Case B — R3 133k clearly improves over R2 80k

Examples:
- R2 only approaches while R3 reaches grasp/lift;
- R3 progresses at least one meaningful physical stage farther.

Outcome:

```text
FULL60_133K_ROBOT_SIGNAL
```

Action:

```text
run one additional A confirmation for R2 and R3
```

If the ordering repeats, proceed to Stage B.

---

## Case C — R2 and R3 are similar / ambiguous

Outcome:

```text
FULL60_80K_133K_AMBIGUOUS
```

Action:

```text
one additional paired A attempt:
R2
R3
```

Do not repeat P1/New-GL automatically.

After two paired observations, decide whether the difference is meaningful.

---

# 11. Stage B — Only If 133k Shows a Real Signal

Stage B is conditional.

If R3 shows confirmed A-point improvement over R2:

test:

```text
Position B:
R1 Train36 New-GL
R2 Full60 80k
R3 Full60 133k

Position C:
R1 Train36 New-GL
R2 Full60 80k
R3 Full60 133k
```

P1-Hist is not required in B/C unless the operator wants one additional practical reference.

Primary Stage B question:

> Does the 133k improvement generalize beyond Position A and approach the Train36 New-GL reference?

---

# 12. Maximum Trial Budget

Expected:

```text
Stage A initial = 4
paired A confirmation = up to 2
Stage B = up to 6
```

Recommended maximum:

```text
12 valid autonomous attempts
```

Do not mechanically consume all 12.

Stop once the scientific comparison is clear.

---

# 13. Primary Comparisons

## C1 — Full60 training exposure

```text
R2 Full60 80k
vs
R3 Full60 133k
```

This is the central R5.11 comparison.

Important caveat:

The R5.3 133k checkpoint belongs to a specific historical trajectory.

Do not interpret differences as a pure step-count causal estimate across all possible runs.

The robot question is practical:

> Is this existing 133k artifact better?

---

## C2 — Full60 vs Train36 strong policy

```text
R3 Full60 133k
vs
R1 Train36 New-GL
```

Question:

> Has the Full60 line finally recovered to the practical level of the strongest Train36 model?

---

## C3 — historical practical reference

```text
R0 P1-Hist
vs
R1 / R2 / R3
```

Use only as a capability anchor.

Do not attribute differences solely to dataset size because architecture/state schema differs.

---

# 14. Primary Real-Robot Metrics

For each model/position record counts/stages:

```text
target-directed approach
valid grasp geometry
close attempt
physical grasp
lift
hold
place/release
autonomous full completion
assisted recovery
```

Do not overinterpret percentages with this small n.

Report:

```text
P4 = 1/2
P6 = 0/2
```

rather than population success claims.

---

# 15. Failure Labels

Assign one primary failure label per valid failed autonomous attempt:

```text
APPROACH_DIRECTION_ERROR
APPROACH_STALL
PREMATURE_RETREAT
GRASP_GEOMETRY_ERROR
CLOSE_TIMING_ERROR
NO_PHYSICAL_GRASP
LIFT_FAILURE
HOLD_FAILURE
PLACE_FAILURE
RUNTIME_FAILURE
```

This is especially important for R2 vs R3.

---

# 16. Runtime Evidence

Record independently:

```text
mean inference latency
p95 latency
max latency
overrun fraction
runtime exception
device error
```

R5.10 already showed roughly stable 17–19 ms inference.

If R5.11 remains similar, do not use generic "compute lag" as an explanation for behavior differences.

---

# 17. Video / CSV Evidence Rule

Continue the established rule:

```text
video and CSV are not assumed synchronized
```

Use:

```text
video:
physical semantic stages

CSV:
state/action/runtime behavior
```

Do not force frame-row alignment.

---

# 18. Formal Outcomes

## `FULL60_133K_ROBOT_RECOVERY_SUPPORTED`

Use if R3 consistently progresses farther than R2 and approaches/equals R1 on meaningful physical stages.

---

## `FULL60_133K_PARTIAL_ROBOT_GAIN`

Use if R3 improves some stages over R2 but remains clearly below R1.

---

## `FULL60_133K_ROBOT_NO_GAIN`

Use if R3 is similar to or worse than R2.

Interpretation:

> longer training of this trajectory does not solve the Full60 real-robot problem.

---

## `TRAIN36_REFERENCE_STILL_DOMINANT`

Use if R1 remains clearly stronger than both Full60 variants.

---

## `P1_REFERENCE_STILL_DOMINANT`

Use descriptively if P1-Hist remains the strongest practical policy.

Do not use this to infer a single causal mechanism.

---

## `OFFLINE_REAL_ROBOT_MISMATCH_CONFIRMED`

Use if the offline ordering fails to predict the robot ordering in a clear way.

---

# 19. What Must NOT Be Done

Do not during R5.11:

```text
retrain 133k
train a new 133k seed
change clipping
change SWA
run SAM
change object conditioning
change normalization
collect new demonstrations
change the task
```

R5.11 is model selection / real-robot diagnosis only.

---

# 20. Required Result Package

Recommended:

```text
experiments/force/r5_11_real_robot_133k_breakthrough_diagnostic/index1/result/
```

Required:

```text
summary.md
validation.md
model_identity_manifest.csv
trial_manifest.csv
initial_state_log.csv
stage_outcomes.csv
failure_labels.csv
runtime_latency.csv
assisted_recovery_log.csv
video_inventory.csv
offline_vs_robot_comparison.csv
builder_review.md
red_team_review.md
proposal.md
videos/
csv/
```

---

# 21. Completion Criteria

`R5_11_COMPLETE = true` requires:

1. exact R0/R1/R2/R3 checkpoint identities resolved;
2. R3 verified as the original R5.3 133k artifact;
3. all Stage-A four models attempted at Position A;
4. P0–P6 stage labels assigned;
5. R2 vs R3 comparison assigned;
6. confirmation trials used only if needed;
7. Stage B executed only if R3 shows real A-point signal;
8. runtime analyzed independently;
9. video/CSV not forcibly synchronized;
10. one formal outcome assigned;
11. Builder Review complete;
12. Red-Team Review complete.

---

# 22. Builder Review Questions

1. Was the exact old R5.3 133k checkpoint used?
2. Was no retraining performed?
3. Was R5.10 M1 reused exactly for the 80k reference?
4. Were P1-Hist and Train36 New-GL treated as capability references rather than clean data-size controls?
5. Was Stage B conditional rather than automatic?
6. Were physical stages actually reviewed from video?
7. Were invalid setup and autonomous failure separated?
8. Was no offline metric substituted for robot outcome?

---

# 23. Red-Team Review Questions

1. Does 133k really behave differently from 80k?
2. Is any apparent 133k gain reproducible on the second A attempt?
3. Does it extend to B/C?
4. Does it approach Train36 New-GL?
5. Could the difference be caused by reset/setup rather than policy?
6. Does runtime remain comparable?
7. If 133k fails, is there any reason left to continue increasing training steps?
8. If Train36 remains superior, should the project move to final-data redesign rather than more optimizer tuning?

---

# 24. Plain-Language Summary

The next real-robot shortlist is:

```text
R0 = P1-Hist
R1 = Train36 New-GL
R2 = Full60 clip10 seed3 / 80k
R3 = the already-existing R5.3 Full60 seed3 / 133k
```

No new model is trained.

First:

```text
all four models
→ Position A
→ one valid attempt each
```

The most important comparison is:

```text
Full60 80k
vs
Full60 133k
```

If 133k clearly looks better, confirm once and then test B/C against Train36 New-GL.

If 133k does not improve, stop spending time on longer Full60 training and reconsider the data/task representation instead.
