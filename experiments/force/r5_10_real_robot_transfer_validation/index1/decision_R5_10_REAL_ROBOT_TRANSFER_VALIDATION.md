# decision_R5_10_REAL_ROBOT_TRANSFER_VALIDATION.md

## 0. Contract Header

- Experiment ID: `R5_10_REAL_ROBOT_TRANSFER_VALIDATION`
- Revision: `r0`
- Mode: `MINIMAL_REAL_ROBOT_TRANSFER_VALIDATION`
- Status: `FROZEN / NOT_RUN`
- New training: `NO`
- New architecture: `NO`
- New dataset collection: `NO`
- Primary purpose:
  - verify whether offline improvements from deterministic training / clip5 / SWA transfer to real closed-loop behavior
- Parent evidence:
  - R5.8: clip5 reduces seed variance but hurts some good seeds
  - R5.9: clip10 SWA is harmful; clip5 SWA has partial offline benefit, especially diagnostic 60k+80k averaging
- Core principle:
  - offline metrics are model-selection evidence, not final task-success evidence

Recommended repository path:

```text
experiments/force/r5_10_real_robot_transfer_validation/index1/decision.md
```

---

# 1. Core Research Question

> Do the offline-stabilized candidates actually improve real-robot approach, grasp, lift, hold, and place behavior?

The round must distinguish:

```text
offline metric improvement
vs
real closed-loop task improvement
```

---

# 2. Frozen Model Set

Use only three primary models.

## M1 — Strong single-checkpoint reference

```text
clip10 / seed3 / 80k
```

Reason:
- strongest controlled clip10 seed;
- historical reference for a good single checkpoint;
- no SWA.

## M2 — Clip5 stabilized single checkpoint

```text
clip5 / seed2 / 80k
```

Reason:
- seed2 was the pathological seed under clip10;
- clip5 produced a large rescue;
- tests whether gradient stabilization transfers to the robot.

## M3 — Clip5 + SWA candidate

```text
clip5 / seed2 / SWA(60k,80k)
```

Reason:
- R5.9 diagnostic window improved seed2 over both 60k and 80k constituents;
- tests whether the offline averaging gain transfers to closed-loop behavior.

Do not include clip10 SWA because R5.9 showed it was harmful offline.

---

# 3. Optional Fourth Model

Only if robot time is convenient and the first three are completed cleanly:

```text
M4 = clip5 / seed3 / SWA(60k,80k)
```

Purpose:
- test whether SWA also helps a previously good-ish clip5 seed.

M4 is optional and must not delay the primary M1/M2/M3 comparison.

---

# 4. Test Positions

Use the established three representative cup positions:

```text
A
B
C
```

Do not change:
- cup;
- fill level;
- camera;
- lighting;
- task definition;
- table setup.

---

# 5. Core Trial Count

Primary pass:

```text
M1 × A/B/C = 3 trials
M2 × A/B/C = 3 trials
M3 × A/B/C = 3 trials
```

Total:

```text
9 primary trials
```

This is a diagnostic transfer round, not a formal success-rate paper claim.

---

# 6. Targeted Confirmation Rule

After the 9 primary trials:

Only if one comparison is ambiguous, allow:

```text
up to 3 additional confirmation trials
```

Examples:
- M2 vs M3 differs only once;
- one trial is invalid due to start-state/setup failure;
- one model shows contradictory behavior across positions.

Maximum planned total:

```text
12 trials
```

Do not automatically repeat every condition.

---

# 7. Reset Protocol

Previous R5 A-round had substantial start-state variation.

Therefore every valid trial must begin from the same verified Home protocol.

Before autonomous execution save:

```text
initial joint state
gripper position
load
camera frame
model ID
position ID
```

A trial is valid only if initial robot state is within the pre-frozen tolerance around the canonical Home.

Do not invent new numeric Home values in this decision.

Use the existing verified safe Home configuration.

If outside tolerance:

```text
INVALID_START_STATE
```

Reset and retry; do not count as a model failure.

---

# 8. Deployment Freeze

All models must use the same deployment path.

Freeze:

```text
torch.backends.cudnn.benchmark = False
same camera index
same crop / resize
same RGB ordering
same DINO/object preprocessing
same state ordering
same load field
same action decoding
same temporal ensembling
same control frequency
same action limits
```

Do not change model-specific runtime settings between trials except checkpoint/model weights.

---

# 9. Trial Stages

For each valid autonomous trial label independently:

```text
P0 = target-directed approach
P1 = valid grasp geometry
P2 = close command / closure attempt
P3 = physical grasp established
P4 = cup lifted
P5 = hold achieved
P6 = place / release sequence completed
```

Also record:

```text
AUTONOMOUS_SUCCESS
ASSISTED_RECOVERY
FAILURE
INVALID_SETUP
```

---

# 10. Assisted-Recovery Rule

Human intervention is allowed only after autonomous completion is already clearly impossible.

If the operator intervenes:

```text
ASSISTED_RECOVERY = true
AUTONOMOUS_SUCCESS = false
```

The assisted continuation remains useful diagnostic evidence.

Do not discard it.

Do not count it as autonomous success.

---

# 11. Video / CSV Rule

Video and CSV/runtime logs are independent evidence streams.

Do not force timestamp alignment.

Analyze:

```text
video → semantic task stages / physical behavior
CSV → command / state / runtime behavior
```

Combine only at trial-level interpretation.

---

# 12. Primary Comparisons

## C1 — Good single baseline vs clip5 rescue

```text
M1 vs M2
```

Question:

> Does clip5's offline stabilization produce competitive real behavior?

## C2 — Single clip5 vs SWA clip5

```text
M2 vs M3
```

Question:

> Does the R5.9 SWA gain transfer to the robot?

## C3 — Offline ranking vs robot ranking

Compare offline ordering with:

```text
P0/P1/P3/P4/P6
```

Question:

> Are offline chunk metrics predictive of closed-loop success?

---

# 13. Primary Real-Robot Metrics

For each model report across A/B/C:

```text
target-directed approach count
valid grasp geometry count
physical grasp count
lift count
hold count
full autonomous completion count
assisted-recovery count
```

Because n is small:

```text
report counts, not statistically overinterpreted percentages
```

Example:

```text
lift = 2/3
```

is acceptable.

Do not make population-level success-rate claims.

---

# 14. Position-Specific Failure Labels

For every failure assign one primary stage:

```text
APPROACH_DIRECTION_ERROR
GRASP_GEOMETRY_ERROR
CLOSE_TIMING_ERROR
NO_PHYSICAL_GRASP
LIFT_FAILURE
HOLD_FAILURE
PLACE_FAILURE
RUNTIME_FAILURE
```

Secondary notes are allowed.

---

# 15. Runtime Metrics

Record:

```text
mean inference latency
p95 latency
max latency
control overrun fraction
runtime exception
safety abort
```

Runtime status does not by itself define task success.

A model may fail physically with clean runtime, or succeed despite some latency.

---

# 16. What Would Count as Transfer Support

## `CLIP5_ROBOT_TRANSFER_SUPPORTED`

Use if M2 is at least competitive with M1 on:
- approach;
- grasp geometry;
- lift;
and does not introduce a new systematic failure.

## `SWA_ROBOT_TRANSFER_SUPPORTED`

Use if M3 improves over M2 on meaningful physical stages, especially:
- approach consistency;
- grasp;
- lift;
- completion.

## `OFFLINE_ROBOT_RANKING_MISMATCH`

Use if offline-preferred models perform clearly worse on robot.

This is a valid scientific result.

---

# 17. Stop / Escalation Logic

### Case A — M1 works, M2/M3 fail similarly

Interpretation:

```text
offline stabilization did not transfer
```

Next:
- inspect deployment/state distribution;
- do not promote clip5/SWA.

### Case B — M2 works better than M1

Interpretation:

```text
clip5 transfer supported
```

Next:
- consider clip5 as training recipe candidate.

### Case C — M3 > M2

Interpretation:

```text
SWA transfer supported
```

Next:
- promote clip5+SWA candidate for broader validation.

### Case D — all models fail in same way

Interpretation:

```text
shared deployment / task / dataset limitation
```

Do not blame optimization method.

### Case E — all models succeed similarly

Interpretation:

```text
offline differences may not matter at current robot-test difficulty
```

Next:
- increase position challenge only in a later decision.

---

# 18. No New Training

During R5.10 do not:

```text
retrain
change clip threshold
change SWA window
run SAM
change normalization
collect demonstrations
change task
```

This is a pure transfer-validation round.

---

# 19. Recommended Robot Order

To reduce setup drift:

```text
Position A:
M1 → M2 → M3

Position B:
M1 → M2 → M3

Position C:
M1 → M2 → M3
```

If thermal/runtime drift across repeated model switches becomes a concern, use a balanced model order and document it.

Do not choose order after seeing results.

---

# 20. Required Result Package

Recommended:

```text
experiments/force/r5_10_real_robot_transfer_validation/index1/result/
```

Required:

```text
summary.md
validation.md
trial_manifest.csv
initial_state_audit.csv
stage_outcomes.csv
failure_labels.csv
runtime_latency.csv
assisted_recovery_log.csv
offline_vs_robot_comparison.csv
builder_review.md
red_team_review.md
proposal.md
videos/
csv/
```

---

# 21. Completion Criteria

`R5_10_COMPLETE = true` requires:

1. M1/M2/M3 identities frozen;
2. A/B/C setup frozen;
3. canonical reset enforced;
4. 9 primary valid trials attempted;
5. invalid-start trials excluded and documented;
6. stage labels P0–P6 assigned from video;
7. CSV/runtime analyzed independently;
8. assisted recovery labeled correctly;
9. offline vs robot ranking compared;
10. one transfer outcome assigned;
11. Builder Review complete;
12. Red-Team Review complete.

---

# 22. Plain-Language Summary

R5.9 tells us which models look promising offline.

R5.10 asks the question that actually matters:

```text
Can they really move toward the cup,
grasp it,
lift it,
hold it,
and place it
on the real robot?
```

Only three main models are needed:

```text
M1 = strong single baseline
M2 = clip5 rescued model
M3 = clip5 + SWA candidate
```

Test each at A/B/C once.

This is enough to determine whether the offline stabilization work is transferring to the physical system before investing in more training or more data.
