# decision_F2_2_CONTACT_GATED_SINGLE_FRAME_LOAD.md

## 0. Contract Header

- Experiment ID: `F2_2_CONTACT_GATED_SINGLE_FRAME_LOAD`
- Revision: `r0`
- Mode: `ABLATION / FEASIBILITY`
- Track: `force`
- Status at creation: `FROZEN / NOT_RUN`
- Scientific scope: `EXPLORATORY_ARCHITECTURE_ABLATION`
- Parent evidence:
  - `F2_1_LOAD_TEMPORAL_ENCODING_ABLATION r1`
  - outcome: `SINGLE_FRAME_LOAD_SUFFICIENT`
  - A1 (single-frame load) improved over clean no-load baseline
  - W15 temporal load was not supported
- Historical context:
  - F1P: `PROMISING_LOAD`
  - F1Q: `UNRESOLVED_MATCHING_COVERAGE`
  - F1RM: `UNRESOLVED_RESIDUAL_CONFOUND`

Recommended repository path:

```text
experiments/force/f2_2_contact_gated_single_frame_load/index1/decision.md
```

This decision does not modify F2.1-r1.
It starts a new experiment round.

---

## 1. Single Research Question

This round answers only:

> Given that single-frame `gripper.load` is useful, is it better to expose that signal to the global OC-ACT policy at all times, or to gate / suppress its contribution outside grasp/contact-relevant states?

This round does **not** answer:
- whether temporal load history helps;
- whether `current` helps;
- whether local residual gripper control is better than global fusion;
- whether higher-frequency local control helps;
- whether real-robot success improves;
- whether `load` is calibrated physical force.

---

## 2. Frozen Baseline from F2.1

Reference architecture:

```text
A1 from F2.1-r1:
6D proprioception
+
single-frame load_t
+
OC-ACT
```

No `current`.

No W15 history.

No temporal encoder.

No local residual policy.

This architecture is the starting point for both F2.2 variants.

---

## 3. Primary Variants

Exactly two primary variants are allowed.

### B0 — Always-On Single-Frame Load

Use the F2.1 A1 semantics:

```text
6D proprioception
+
load_t
↓
OC-ACT
```

The load feature is always available to the policy.

No gating.

---

### B1 — Contact-Gated Single-Frame Load

Use the same:

```text
6D proprioception
+
load_t
```

but modulate only the load representation by a causal gate:

```text
load_feature_gated = gate_t * load_feature
```

where:

```text
gate_t ∈ [0, 1]
```

or a binary equivalent if the chosen implementation is deterministic.

All other inputs and policy structure must remain matched to B0.

---

## 4. Scientific Contrast

Primary contrast:

```text
B1 vs B0
```

Question:

> Does contact-aware gating improve how the policy uses actuator load?

The round must not be reframed into:
- gated vs no-load;
- gated vs W15;
- gated vs local residual control.

Those require separate experiments.

---

## 5. Gate Design Principle

The gate must be:

```text
causal
simple
interpretable
small-capacity
```

It must not become a hidden full policy.

Its role is only:

> Estimate whether current robot/gripper state is plausibly within a grasp/contact-relevant regime, and scale the load feature accordingly.

---

## 6. Gate Inputs

The gate may use only causal variables that are already available online.

Allowed primary inputs:

```text
gripper.pos
P1 / policy gripper command or target
delta gripper.pos
delta gripper command
```

Optional:

```text
single-frame load_t
```

only if the exact gate design is frozen before training and the Red-Team review explicitly checks for circularity / phase shortcut risk.

Preferred first implementation:

```text
gate from non-load gripper state/action variables
```

to keep the interpretation clean.

Forbidden:

```text
future frames
manual phase labels at inference
episode progress
absolute frame number
Dev/Test labels
ground-truth contact labels at inference
current/curr
W15 history
```

---

## 7. Gate Supervision

Two implementation styles are allowed.

### Option G1 — Deterministic Gate

A simple predefined rule from online gripper state/action.

Example structure:

```text
closing-or-closed
AND
gripper.pos within grasp-relevant region
```

No learned gate.

### Option G2 — Learned Small Gate

A tiny classifier / MLP trained on Train-side phase/contact-related supervision that is already authorized and independently defined.

If G2 is used:

- labels must not be generated from `load`;
- labels must not use `current`;
- Dev12 must not be used to train the gate;
- the gate model capacity must remain small;
- gate training budget and metrics must be reported separately.

Only one of G1 or G2 may be selected for r0.
The executor must freeze the choice before main training.

Do not compare G1 vs G2 within this decision.

---

## 8. Recommended r0 Gate

Preferred implementation:

```text
G1 deterministic gate
```

Reason:

- keeps F2.2 focused on the value of gating itself;
- avoids introducing a second learned model;
- reduces confounds;
- is easier to interpret in Builder/Red-Team review.

The exact threshold / gripper region must be frozen from:
- Train36 statistics;
- existing project semantics;
- existing grasp behavior.

Do not tune it on Dev12 after seeing policy performance.

---

## 9. Gate Behavior

Outside the gate:

```text
load_feature_gated = 0
```

or an equivalent normalized neutral baseline.

Inside the gate:

```text
load_feature_gated = load_feature
```

For soft gating:

```text
load_feature_gated = gate_t * load_feature
```

No other feature may be gated.

The gate must not directly alter actions.

---

## 10. Common Architecture Constraint

B0 and B1 must share:

```text
same 6D proprioception
same RGB/object-conditioning pathway
same OC-ACT backbone
same single-frame load encoder
same load normalization
same load fusion point
same action space
same chunking
same CVAE settings
same temporal ensembling
same action normalization
same training budget
```

Only authorized difference:

```text
B1 gates the load feature
B0 does not
```

---

## 11. Initialization Strategy

Preferred clean design:

```text
B0 and B1 trained under the same initialization protocol from scratch / common backbone initialization
```

Do not warm-start only B1 from the F2.1 A1 checkpoint.

If compute constraints motivate warm-starting both variants from the same A1 checkpoint, stop and raise a proposal because that changes the experimental semantics.

The selected strategy must be frozen before training.

---

## 12. Dataset and Split

Use the same frozen dataset/split as F2.1-r1:

```text
Train36
Dev12
Final12 sealed
```

Rules:

- Train36 for training;
- Dev12 for frozen offline ablation;
- Final12 untouched;
- no re-splitting;
- no new robot data;
- no F1RM targeted teleop injection unless already part of Train36 by prior contract.

---

## 13. Load Representation

Use only:

```text
single-frame normalized load_t
```

No:

```text
W15
moving average
slope
delta-load temporal features
current
```

This preserves the F2.1 conclusion that single-frame load is the supported representation.

---

## 14. Primary Metric

Use the same frozen primary metric as F2.1-r1:

```text
Dev12 normalized h0 gripper L1
```

Lower is better.

Do not change the metric after results are known.

---

## 15. Secondary Metrics

Report:

```text
Dev12 denormalized h0 gripper MAE
Dev12 arm action MAE
overall action error if established
per-episode primary metric
training health
latency
parameter count
VRAM if measurable
```

For B1 additionally report gate behavior:

```text
gate active fraction
gate active fraction per episode
gate activation distribution over gripper position
gate activation distribution over gripper command
```

---

## 16. Mechanism Diagnostics

Required where technically feasible.

### M1 — Gate-off counterfactual

For B1, force:

```text
gate = 0
```

on Dev observations and compare predictions.

Purpose:

> Determine whether gated load meaningfully affects behavior.

### M2 — Gate-on counterfactual

For B1, force:

```text
gate = 1
```

for all frames.

Compare to normal B1 behavior.

Purpose:

> Determine whether the learned/trained policy depends on selective gating versus always-on load exposure.

### M3 — Load-zero counterfactual

Set the load feature to neutral baseline.

Report gripper and arm prediction changes.

### M4 — Per-episode gate activity

Check whether the gate is:
- always on;
- almost never on;
- concentrated in plausible grasp-related regions.

If the gate degenerates to nearly constant behavior, this must be reported.

---

## 17. Gate Degeneracy Rules

Flag:

```text
GATE_DEGENERATE_ALWAYS_ON
```

if the gate is active for an overwhelming fraction of frames such that it effectively reproduces B0.

Flag:

```text
GATE_DEGENERATE_ALWAYS_OFF
```

if it rarely activates and the policy effectively ignores load.

Exact descriptive thresholds may be reported, but no post-hoc threshold tuning is allowed.

---

## 18. Main Outcome Categories

### `CONTACT_GATING_PROMISING`

Use only if:

1. B1 improves over B0 on the frozen primary metric;
2. improvement is directionally consistent across a meaningful fraction of Dev episodes;
3. arm-action metrics do not materially degrade;
4. the gate is non-degenerate;
5. mechanism diagnostics show selective load usage;
6. compute overhead is acceptable.

Interpretation:

> Selective contact/grasp-aware use of actuator load is justified for the next architecture stage.

---

### `ALWAYS_ON_LOAD_SUFFICIENT`

Use if:

- B0 is equal or better than B1;
- B1 gate is non-degenerate but provides no meaningful stable benefit.

Interpretation:

> Contact gating is not justified; retain the simpler always-on single-frame load architecture.

---

### `GATING_HURTS`

Use if:

- B1 consistently worsens gripper prediction and/or destabilizes arm prediction.

Interpretation:

> The chosen gating mechanism should not be carried forward.

---

### `GATE_DEGENERATE`

Use if:

- gate is effectively always-on or always-off;
- the experiment therefore does not meaningfully test selective load use.

---

### `UNRESOLVED`

Use if:

- implementation differs beyond gating;
- training is unhealthy;
- metric reproduction fails;
- split leakage occurs;
- baseline identity is unclear;
- the gate semantics are not reproducible.

---

## 19. Real-Robot Boundary

No comparative real-robot effectiveness test is authorized.

Allowed:

```text
offline training
Dev12 evaluation
mechanism diagnostics
deployment-cost measurement
```

Optional:

```text
checkpoint loadability / inference smoke test
```

on the user's RTX 4060 8GB laptop.

No task-success claim.

---

## 20. No Local Residual Policy

Do not add:

```text
delta gripper action
downstream local controller
high-frequency regulator
```

This round remains global OC-ACT fusion.

Global-vs-local decomposition is reserved for F2.3.

---

## 21. No Post-Hoc Rescue

After results are known, do not:

- add W15;
- add current;
- change the gate threshold;
- switch deterministic gate to learned gate;
- switch learned gate to deterministic gate;
- alter the load fusion point;
- add FiLM;
- add local residual policy;
- train longer only for B1;
- change the primary metric;
- use Final12;
- add real-robot success as rescue evidence.

Any meaningful change requires a new revision / experiment.

---

## 22. Completion Criteria

`F2_2_COMPLETE = true` requires:

1. B0/B1 semantics verified;
2. exact gate definition frozen;
3. same load representation used;
4. same common architecture used;
5. same Train36/Dev12 split used;
6. Final12 untouched;
7. matched training budget verified;
8. frozen primary metric reported;
9. per-episode metrics reported;
10. gate activity audit complete;
11. gate counterfactual diagnostics complete where feasible;
12. implementation-difference audit complete;
13. latency/parameter cost reported;
14. outcome category assigned;
15. result package complete.

Completion does not imply gating works.

---

## 23. Expected Result Package

Recommended path:

```text
experiments/force/f2_2_contact_gated_single_frame_load/index1/result/
```

Required:

```text
summary.md
validation.md
baseline_identity.json
gate_config.json
variant_configs/
training_manifest.csv
training_health.md
metrics.json
per_episode_metrics.csv
gate_activity.csv
gate_counterfactuals.csv
mechanism_review.md
implementation_diff.md
latency_and_params.json
leakage_audit.md
figures/
proposal.md
```

Recommended figures:

```text
figures/b0_b1_primary_metric.png
figures/per_episode_primary_metric.png
figures/gate_activity_timeline.png
figures/gate_activity_vs_gripper_position.png
figures/gate_activity_vs_command.png
figures/gate_counterfactual_effect.png
figures/arm_mae_comparison.png
figures/training_curves.png
```

---

## 24. Builder Review Questions

Builder Review must verify:

1. Are B0 and B1 identical except for gating?
2. Is single-frame load the only actuator-feedback input?
3. Is current absent?
4. Is W15 absent?
5. Is the gate causal?
6. Is the gate definition reproducible?
7. Was the gate frozen before seeing Dev performance?
8. Are training budgets matched?
9. Is Final12 untouched?
10. Is the primary metric unchanged from F2.1?
11. Is the gate non-action-producing?
12. Are deployment costs acceptable?

---

## 25. Red-Team Review Questions

Red-Team Review must ask:

1. Is the gate merely a hand-coded phase detector?
2. Does the gate improve results because it suppresses noisy load, or because it changes effective model capacity?
3. Is gate activity concentrated in only a few episodes?
4. Does B1 improve gripper prediction while harming arm prediction?
5. Is the gate effectively always-on / always-off?
6. Could the same improvement be obtained by removing load entirely outside a broad gripper-position range?
7. Are gate thresholds post-hoc tuned?
8. Does this establish only selective load use, not contact sensing or force control?

---

## 26. Plain-Language Summary

F2.1 showed:

```text
single-frame load helps
W15 temporal load does not
```

So this round keeps the supported simple representation and asks a new question:

```text
Should the model see load all the time,
or mainly during grasp/contact-relevant moments?
```

Two models:

```text
B0:
single-frame load always active

B1:
same single-frame load,
but gated outside grasp/contact-relevant states
```

Nothing else changes.

If B1 wins, selective use of load is worth carrying forward.

If B0 is just as good or better, keep the simpler always-on design.

---

## 27. Handoff Rule

After execution:

1. stop;
2. return the full result package;
3. perform Builder Review;
4. perform Red-Team Review;
5. discuss with the user;
6. do not automatically start F2.3;
7. do not add local residual policy;
8. do not unseal Final12.

Only if F2.2 justifies it may a later decision compare:

```text
global fused load
vs
local residual gripper policy
```

That future round is not authorized here.
