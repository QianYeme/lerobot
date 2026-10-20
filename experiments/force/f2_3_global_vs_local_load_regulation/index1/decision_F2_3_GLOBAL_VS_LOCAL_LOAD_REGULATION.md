# decision_F2_3_GLOBAL_VS_LOCAL_LOAD_REGULATION.md

## 0. Contract Header

- Experiment ID: `F2_3_GLOBAL_VS_LOCAL_LOAD_REGULATION`
- Revision: `r0`
- Mode: `ABLATION / FEASIBILITY`
- Track: `force`
- Status at creation: `FROZEN / NOT_RUN`
- Scientific scope: `EXPLORATORY_ARCHITECTURE_ABLATION`
- Parent evidence:
  - `F2_1_LOAD_TEMPORAL_ENCODING_ABLATION r1`
    - outcome: `SINGLE_FRAME_LOAD_SUFFICIENT`
  - `F2_2_CONTACT_GATED_SINGLE_FRAME_LOAD`
    - outcome: `GATING_HURTS`
- Historical evidence:
  - F1P: load showed exploratory incremental information
  - F1Q/F1RM: contact-related evidence remained confounded by actuator state
- Current best representation:
  - single-frame `gripper.load`
  - always-on
  - no `current`
  - no W15 temporal encoder
  - no deterministic contact gate

Recommended repository path:

```text
experiments/force/f2_3_global_vs_local_load_regulation/index1/decision.md
```

This is a new experiment contract.
It does not overwrite F2.1 or F2.2.

---

## 1. Single Research Question

This round answers two tightly related questions:

### Primary question

> Is actuator load better used inside the global OC-ACT policy, or as an input to a separate local residual gripper policy?

### Secondary question

> If a local residual gripper policy helps, is that improvement specifically due to `gripper.load`, or would a small local correction network help even without load?

This round does **not** answer:

- whether W15 history helps;
- whether `current` helps;
- whether learned contact gating helps;
- whether high-frequency local control helps;
- whether force control is calibrated in Newtons;
- whether final real-robot task success is improved.

---

## 2. Frozen Variants

Exactly three primary variants are allowed.

### C0 — Global Load Fusion

Use the current supported architecture from F2.1/F2.2:

```text
RGB
+
6D proprioception
+
single-frame load_t
↓
global OC-ACT
↓
arm actions + gripper action
```

`load` is fused inside the global policy.

No local residual module.

---

### C1-no-load — Local Residual Without Load

Global policy:

```text
RGB
+
6D proprioception
↓
global OC-ACT
↓
arm actions + nominal gripper action
```

The global policy receives:

```text
no load
no current
```

A separate local residual policy receives:

```text
gripper.pos_t
nominal_gripper_t
```

and predicts:

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

Purpose:

> Control for the possibility that a small local correction network helps even without actuator load.

---

### C1-load — Local Residual With Load

Global policy is identical to C1-no-load:

```text
RGB
+
6D proprioception
↓
global OC-ACT
```

The local residual policy receives:

```text
load_t
gripper.pos_t
nominal_gripper_t
```

and predicts:

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

This is the main local load-aware architecture.

---

## 3. Frozen Scientific Contrasts

### Primary contrast

```text
C1-load vs C0
```

Question:

> Is global-local decomposition better than global load fusion?

### Secondary contrast

```text
C1-load vs C1-no-load
```

Question:

> Does load add value inside the local residual branch?

### Contextual contrast

```text
C1-no-load vs C0
```

Question:

> Does local correction itself help even without load?

Do not redefine the primary contrast after results are known.

---

## 4. Global Policy Semantics

### C0 global policy

Use:

```text
6D proprioception
+
single-frame load
```

inside the global OC-ACT backbone.

### C1 global policy

Use:

```text
6D proprioception only
```

The C1 global policy must not receive:

```text
load
current
W15 history
```

This separation is required for a clean global-vs-local comparison.

---

## 5. Global Policy Freezing

For C1-no-load and C1-load:

```text
global OC-ACT is frozen
```

during local residual training.

Only the local residual MLP is trainable.

Reason:

> If the global policy is fine-tuned jointly, any C1 improvement could come from global re-adaptation rather than the local decomposition itself.

---

## 6. Global Checkpoint Identity

Use the same verified 6D no-actuator-feedback OC-ACT checkpoint for:

```text
C1-no-load
C1-load
```

Preferred source:

```text
F2.1 A0-compatible global model
```

The executor must record:

```text
checkpoint ID/path
git HEAD
training config
dataset/split
primary metric
```

If a valid common 6D global checkpoint cannot be established, stop:

```text
BLOCKED_GLOBAL_CHECKPOINT_IDENTITY
```

Do not silently substitute another model.

---

## 7. C0 Identity

C0 should use the supported always-on single-frame load architecture.

Preferred reference:

```text
F2.1 A1 / F2.2 B0 semantics
```

If an existing validated checkpoint is reused, record it.

If C0 is retrained, its training protocol and seed policy must be documented.

Do not change C0 architecture in this round.

---

## 8. Residual Target Construction

For every Train36 frame:

1. run the frozen 6D global OC-ACT under teacher/expert observation replay;
2. obtain:

```text
nominal_gripper_t
```

3. read the demonstrated gripper action:

```text
demo_gripper_t
```

4. define raw residual:

```text
raw_residual_t
=
demo_gripper_t
-
nominal_gripper_t
```

5. define bounded target:

```text
residual_target_t
=
clip(
    raw_residual_t,
    -delta_max,
    +delta_max
)
```

This target is used for C1-no-load and C1-load.

---

## 9. Residual Bound

Freeze:

```text
delta_max
=
5% of the demonstrated gripper action range
```

The exact native-unit value must be computed from Train36 only and written to:

```text
result/residual_config.json
```

The bound may be made smaller only for documented safety/deployment reasons before training.

It may not be increased after results are known.

---

## 10. Residual Saturation Audit

Before training, report:

```text
fraction of raw residual targets clipped at +delta_max
fraction clipped at -delta_max
total saturation fraction
```

During evaluation, also report:

```text
fraction of predicted residuals near ±delta_max
```

A model that spends most of its time saturated should not be interpreted as a subtle local regulator.

---

## 11. Local Policy Architecture

Use a small MLP.

Preferred r0 structure:

```text
input
↓
Linear
↓
nonlinearity
↓
Linear
↓
nonlinearity
↓
Linear
↓
tanh
↓
delta_gripper
```

The final `tanh` output is scaled by:

```text
delta_max
```

The MLP must remain lightweight.

Do not use:

```text
Transformer
GRU
W15 temporal encoder
CNN
vision features
object tokens
```

in r0.

---

## 12. Local Inputs

### C1-no-load

Exactly:

```text
gripper.pos_t
nominal_gripper_t
```

### C1-load

Exactly:

```text
normalized load_t
gripper.pos_t
nominal_gripper_t
```

No:

```text
current
delta-load
W15 history
RGB
arm joints
manual phase labels
episode progress
```

unless a later revision authorizes them.

---

## 13. Normalization

Fit input normalization using Train36 only.

For C1-load:

```text
load normalization
```

must match the semantics used in F2.1/F2.2 where possible.

For both local policies:

```text
gripper.pos
nominal_gripper
residual_target
```

normalization must be documented.

Do not fit any normalization using Dev12 or Final12.

---

## 14. Training Dataset

Use:

```text
Train36
```

only for local-policy training.

Dev12 is reserved for frozen offline evaluation.

Final12 remains sealed.

No new robot data collection is authorized.

Do not inject F1RM targeted teleop trials unless they are already formally part of Train36.

---

## 15. Training Budget

C1-no-load and C1-load must have:

```text
same optimizer
same learning rate
same batch size
same training steps/epochs
same seed policy
same checkpoint-selection rule
```

The global policy remains frozen.

Do not train C1-load longer because it has one more input.

---

## 16. Primary Metric

Keep continuity with F2.1/F2.2.

Primary metric:

```text
Dev12 normalized h0 gripper L1
```

Lower is better.

Do not redefine the primary metric after results are seen.

---

## 17. Secondary Metrics

Report:

```text
Dev12 denormalized h0 gripper MAE
Dev12 arm action MAE
overall action error if already established
per-episode primary metric
training health
latency
parameter count
VRAM if measurable
```

For local policies additionally report:

```text
mean |delta_gripper|
median |delta_gripper|
max |delta_gripper|
near-bound fraction
residual sign distribution
per-episode residual magnitude
```

---

## 18. Arm Preservation Requirement

For C1-no-load and C1-load:

```text
arm actions must come directly from the same frozen 6D global policy
```

The local module must not modify arm actions.

Therefore:

```text
arm predictions for C1-no-load and C1-load
```

must be numerically identical under the same observations, except for serialization precision.

If they differ materially:

```text
INVALID_LOCAL_ISOLATION
```

---

## 19. Offline Evaluation Procedure

All three variants are evaluated on the same Dev12 expert-observation replay protocol.

For C1 variants:

1. global 6D policy predicts nominal action;
2. local MLP predicts residual;
3. final gripper prediction is:

```text
nominal + residual
```

4. arm prediction remains nominal.

Do not run real-robot closed-loop effectiveness testing in r0.

---

## 20. Mechanism Diagnostics

Required:

### M1 — Zero-residual counterfactual

For C1 variants:

```text
delta_gripper = 0
```

This must reproduce the frozen 6D global baseline gripper prediction.

### M2 — Load-zero counterfactual

For C1-load:

replace normalized load with its neutral baseline and report the change in:

```text
predicted residual
final gripper prediction
```

Purpose:

> Verify that load is actually used by the local branch.

### M3 — Residual-target correlation

Compare:

```text
predicted residual
vs
bounded residual target
```

on Dev12.

### M4 — Episode-level residual behavior

Report whether residual corrections are:
- consistently small;
- concentrated in a few episodes;
- dominated by one sign;
- frequently saturated.

---

## 21. Capacity Audit

Report:

```text
C0 total parameters
C1 global parameters
C1 local parameters
C1 trainable parameters
latency overhead
```

A key architectural distinction is:

```text
C0:
global policy trainable with load fusion

C1:
global policy frozen
small local branch trainable
```

Do not hide this asymmetry.

---

## 22. Main Outcome Categories

### `LOCAL_LOAD_REGULATION_PROMISING`

Use only if all are true:

1. C1-load improves over C0 on the frozen primary metric;
2. C1-load improves over C1-no-load;
3. the improvement is directionally consistent across a meaningful fraction of Dev episodes;
4. arm actions are preserved exactly in C1;
5. residual corrections remain predominantly within a small non-saturated regime;
6. load-zero diagnostics show meaningful use of load;
7. latency/compute overhead remains acceptable.

Interpretation:

> Actuator load is better supported as an input to a local gripper residual regulator than as a globally fused OC-ACT feature.

---

### `LOCAL_RESIDUAL_HELPS_BUT_LOAD_NOT_NEEDED`

Use if:

- C1-no-load improves over C0;
- C1-load does not materially outperform C1-no-load.

Interpretation:

> The local correction structure helps, but current evidence does not show that load is the reason.

---

### `GLOBAL_FUSION_REMAINS_BEST`

Use if:

- C0 is equal or better than both local variants.

Interpretation:

> Keep single-frame load inside the global OC-ACT policy; local decomposition is not justified by this experiment.

---

### `LOCAL_LOAD_HELPS_BUT_NOT_OVER_GLOBAL`

Use if:

- C1-load improves over C1-no-load;
- but does not beat C0.

Interpretation:

> Load is useful locally, but global fusion remains the stronger architecture.

---

### `LOCAL_POLICY_SATURATES`

Use if:

- C1 depends heavily on ±delta_max clipping;
- residual behavior no longer resembles small correction.

Interpretation:

> The residual formulation or bound is not validated.

---

### `UNRESOLVED`

Use if:

- checkpoint identity is uncertain;
- local/global semantics differ beyond contract;
- training health fails;
- metric reproduction fails;
- leakage occurs;
- residual construction is invalid.

---

## 23. Real-Robot Boundary

This round authorizes:

```text
offline residual-target generation
local-policy training
Dev12 replay evaluation
mechanism diagnostics
latency/parameter analysis
```

It does not authorize:

```text
real-robot comparative effectiveness testing
automatic gripper correction on hardware
Final12 inference
```

A later experiment may test C1 on hardware only after Builder + Red-Team review.

---

## 24. No Post-Hoc Rescue

After results are known, do not:

- increase delta_max above 5%;
- add W15;
- add current;
- add a contact gate;
- add more local inputs;
- switch MLP to GRU/Transformer;
- fine-tune the global C1 policy;
- change C0 architecture;
- change the primary metric;
- selectively add training steps;
- unseal Final12;
- add real-robot performance as rescue evidence.

Any meaningful change requires a new revision / experiment.

---

## 25. Completion Criteria

`F2_3_COMPLETE = true` requires:

1. C0 identity verified;
2. common 6D frozen global checkpoint verified for both C1 variants;
3. local residual targets generated correctly;
4. 5% residual bound frozen from Train36;
5. residual saturation audit complete;
6. C1-no-load and C1-load trained under matched budgets;
7. Dev12 evaluated under the same protocol;
8. primary metric reported;
9. per-episode metrics reported;
10. arm-preservation audit passed;
11. zero-residual and load-zero diagnostics complete;
12. residual magnitude/saturation metrics reported;
13. latency/parameter audit complete;
14. Final12 untouched;
15. outcome category assigned;
16. result package complete.

Completion does not imply local regulation works.

---

## 26. Expected Result Package

Recommended path:

```text
experiments/force/f2_3_global_vs_local_load_regulation/index1/result/
```

Required:

```text
summary.md
validation.md
baseline_identity.json
global_checkpoint_identity.json
residual_config.json
residual_target_stats.json
variant_configs/
training_manifest.csv
training_health.md
metrics.json
per_episode_metrics.csv
residual_metrics.csv
load_zero_counterfactual.csv
zero_residual_counterfactual.csv
arm_preservation_audit.md
mechanism_review.md
implementation_diff.md
latency_and_params.json
leakage_audit.md
figures/
proposal.md
```

Recommended figures:

```text
figures/c0_c1_primary_metric.png
figures/per_episode_primary_metric.png
figures/residual_target_distribution.png
figures/predicted_vs_target_residual.png
figures/residual_magnitude_per_episode.png
figures/load_zero_effect.png
figures/arm_mae_comparison.png
figures/training_curves.png
```

---

## 27. Builder Review Questions

Builder Review must verify:

1. Is C0 the supported always-on single-frame global-load architecture?
2. Do both C1 variants use the same frozen 6D global checkpoint?
3. Does the C1 global policy truly exclude load/current?
4. Is the local branch the only trainable component in C1?
5. Are C1 arm actions unchanged by the local branch?
6. Is the residual target exactly demo minus nominal, then clipped?
7. Is delta_max exactly the frozen 5% Train36 range?
8. Are C1-no-load and C1-load identical except for load input?
9. Are training budgets matched?
10. Is Final12 untouched?
11. Are residual saturation and mechanism diagnostics complete?
12. Are deployment costs acceptable?

---

## 28. Red-Team Review Questions

Red-Team Review must ask:

1. Does C1-load actually beat C1-no-load?
2. Does C1-load actually beat C0?
3. Is any benefit driven by a few Dev episodes?
4. Are residual outputs mostly saturated?
5. Is C1 merely learning a static bias correction?
6. Does load-zero meaningfully change residual outputs?
7. Could C1-no-load explain most of the gain?
8. Is the 5% bound hiding a need for a full gripper policy?
9. Is the comparison fair given C0 is globally trained while C1 global is frozen?
10. Does this establish local actuator-feedback regulation only, not calibrated force control?

---

## 29. Plain-Language Summary

This round compares three ideas:

```text
C0:
load goes into the whole OC-ACT policy

C1-no-load:
the main OC-ACT policy does not see load
a small local MLP only corrects the gripper

C1-load:
same local correction design
but the small MLP also sees load
```

The local policy does not replace P1.

It only predicts:

```text
small delta_gripper
```

and the correction is bounded to:

```text
±5% of the normal demonstrated gripper range
```

This round therefore answers two questions:

```text
Does a local residual gripper policy help?

If yes, is actuator load actually responsible for the extra benefit?
```

If C1-load beats both C0 and C1-no-load, that is strong evidence for a global-task / local-physical-interaction decomposition.

---

## 30. Handoff Rule

After execution:

1. stop;
2. return the complete result package;
3. perform Builder Review;
4. perform Red-Team Review;
5. discuss with the user;
6. do not automatically deploy C1 on the robot;
7. do not unseal Final12;
8. do not automatically add high-frequency control.

Only if F2.3 supports local regulation may a later decision authorize:
- a tiny real-robot feasibility test;
- or a higher-frequency local regulator.

Those are not authorized here.
