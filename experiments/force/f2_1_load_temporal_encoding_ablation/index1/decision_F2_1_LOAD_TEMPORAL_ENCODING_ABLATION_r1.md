# decision_F2_1_LOAD_TEMPORAL_ENCODING_ABLATION_r1.md

## 0. Contract Header

- Experiment ID: `F2_1_LOAD_TEMPORAL_ENCODING_ABLATION`
- Revision: `r1`
- Mode: `ABLATION / FEASIBILITY`
- Track: `force`
- Status at creation: `FROZEN / NOT_RUN`
- Scientific scope: `EXPLORATORY_ARCHITECTURE_ABLATION`
- Supersedes: `F2_1_LOAD_TEMPORAL_ENCODING_ABLATION r0` for execution purposes only
- r0 status remains recorded as:
  - `BLOCKED_INPUT_SEMANTICS`
  - no policy training result
- Parent evidence:
  - `F1P_CONTACT_SIGNAL_PILOT`: `PROMISING_LOAD`
  - `F1Q_LOAD_CONFOUND_CONTROL`: `UNRESOLVED_MATCHING_COVERAGE`
  - `F1RM_TELEOP_LOAD_RETROSPECTIVE_MATCH`: `UNRESOLVED_RESIDUAL_CONFOUND`

Recommended repository path:

```text
experiments/force/f2_1_load_temporal_encoding_ablation/index1/decision.md
```

Recommended archive of prior revision:

```text
experiments/force/f2_1_load_temporal_encoding_ablation/index1/decision_r0.md
```

Do not delete the r0 evidence/result package.

---

## 1. Why r1 Exists

r0 preflight established that the current historical P1 / OC-ACT already receives an 8D robot state containing:

```text
5 arm joint positions
+ gripper.pos
+ gripper.load
+ gripper.current/curr
```

Therefore the original r0 definition:

```text
A0 = P1 without load
A1 = P1 + single-frame load
A2 = P1 + W15 load
```

was scientifically invalid.

r1 fixes that problem by constructing a clean common proprioceptive baseline that excludes both actuator-feedback channels from all variants before adding load back in a controlled way.

---

## 2. Single Research Question

This round answers:

> Under a common OC-ACT backbone and a common 6D proprioceptive baseline, does a causal W15 history of `gripper.load` provide more useful gripper-action information than a single current `gripper.load` value?

Secondary question:

> Does adding single-frame `gripper.load` provide useful information relative to no actuator-feedback input?

This round does **not** answer:

- whether `current` is useful;
- whether contact gating helps;
- whether a local residual gripper policy helps;
- whether higher-frequency local control helps;
- whether load-aware control improves final real-robot task success;
- whether `load` is calibrated force.

---

## 3. Historical P1 Reference

Keep the current accepted P1 / OC-ACT model as a historical reference:

```text
P_hist:
current P1 / OC-ACT
8D state
includes load + current
```

`P_hist` is **not** one of the three primary causal ablation variants.

It may be reported as contextual reference only.

No conclusion about A0/A1/A2 may rely on P_hist replacing one of the frozen contrasts.

---

## 4. Frozen Primary Variants

Exactly three primary variants are allowed.

### A0 — Clean No-Actuator-Feedback Baseline

Input state:

```text
5 arm joint positions
+ gripper.pos
= 6D proprioception
```

No:

```text
gripper.load
gripper.current/curr
```

No actuator-feedback branch.

---

### A1 — Single-Frame Load

Common 6D proprioception:

```text
5 arm joint positions
+ gripper.pos
```

plus:

```text
load_t
```

No `current`.

No temporal history.

---

### A2 — Temporal Load

Common 6D proprioception:

```text
5 arm joint positions
+ gripper.pos
```

plus causal:

```text
load_(t-14:t)
```

encoded with a small temporal encoder.

No `current`.

No future frames.

---

## 5. Frozen Scientific Contrasts

Primary contrast:

```text
A2 vs A1
```

Question:

> Does temporal load history add value beyond a single load sample?

Secondary contrast:

```text
A1 vs A0
```

Question:

> Does load itself add value beyond position-only proprioception?

Additional reference:

```text
A2 vs A0
```

Context only.

Do not redefine the main contrast after results are known.

---

## 6. Common Backbone Constraint

A0, A1, and A2 must share the same OC-ACT backbone semantics.

Frozen common elements include:

```text
vision branch
object-conditioning branch
ACT backbone
action dimension
chunking semantics
CVAE settings
temporal ensembling settings
action normalization
camera inputs
training dataset
training split
optimizer family
training budget
checkpoint-selection rule
```

Only actuator-feedback representation may differ as explicitly authorized.

---

## 7. State Semantics

All primary variants use the same 6D base state:

```text
[joint_1, joint_2, joint_3, joint_4, joint_5, gripper.pos]
```

The exact field ordering must be frozen and recorded.

The executor must verify that:

```text
load
current/curr
```

are truly removed from the common base state tensor for A0/A1/A2.

Do not merely zero values while preserving hidden preprocessing differences unless explicitly documented and proven equivalent.

---

## 8. Load Normalization

Load normalization must be defined before training.

Rules:

- fit normalization statistics on Train36 only;
- use the same load normalization for A1 and A2;
- do not fit using Dev12;
- do not fit using Final12;
- do not use contact labels to normalize load.

Record:

```text
mean
std
clipping if any
missing-value handling
```

in the result package.

---

## 9. A1 Load Representation

A1 uses only:

```text
normalized load_t
```

Project it through the minimal representation needed to enter the common fusion point.

No history-derived features are allowed.

No:

```text
mean
std
slope
delta
moving average
```

in A1.

---

## 10. A2 Temporal Load Representation

A2 uses causal W15:

```text
load_(t-14:t)
```

Preferred encoder:

```text
small MLP or shallow 1D temporal encoder
```

The encoder must be lightweight enough for the user's RTX 4060 8GB deployment target.

Do not use a large temporal Transformer in r1 unless repository constraints make the smaller options impossible.

Document:

```text
input shape
hidden dimensions
output dimension
parameter count
activation
normalization
```

before training.

---

## 11. Startup Handling

For frames with fewer than 15 historical samples, use one frozen causal rule.

Preferred:

```text
left-pad with the earliest available load value
```

Alternative:

```text
causal available-prefix mask
```

Whichever is selected must be:

- fixed before training;
- identical in Train and Dev;
- documented;
- free of future leakage.

---

## 12. Fusion Point

A1 and A2 must enter the policy at the same fusion point.

Preferred options:

```text
project to one actuator-feedback token
and append at the same token level
```

or, if repository architecture strongly favors state fusion:

```text
project to the same feature dimension
and concatenate at the same state-feature location
```

A1 and A2 may differ only in how load is encoded before that common fusion point.

Do not give A2 privileged access to additional layers.

---

## 13. Current Signal Boundary

`gripper.current/curr` is excluded from all A0/A1/A2 variants.

Do not:

- add current to A2;
- use current as a fallback;
- use current in contact labels;
- use current in checkpoint selection;
- use current in a rescue experiment inside r1.

Future `load vs load+current` ablation requires a new decision.

---

## 14. Dataset and Split

Use the frozen formal dataset:

```text
Train36
Dev12
Final12
```

Rules:

- Train36: training only;
- Dev12: allowed for the frozen offline ablation evaluation;
- Final12: sealed;
- no re-splitting;
- no adjacent-frame random split;
- no new data collection;
- no mixing F1RM targeted teleop trials into policy training unless they are already part of the formal Train36 by prior contract.

If repository reality conflicts with the expected split, stop and report:

```text
BLOCKED_SPLIT_MISMATCH
```

---

## 15. Training Initialization

A0, A1, A2 must use a matched initialization protocol.

Preferred scientific design:

```text
same pretrained visual/object backbone initialization
same ACT initialization source
same random-seed policy
new load-specific layers initialized under the same rule
```

Do not use the current 8D P_hist checkpoint in a way that transfers learned load/current weights only to some variants.

If warm-starting from P_hist is considered necessary for engineering reasons, stop and raise a proposal because that changes the causal interpretation.

---

## 16. Training Budget

Freeze before execution:

```text
optimizer
learning rate
scheduler
batch size
number of optimization steps / epochs
weight decay
gradient clipping
seed(s)
checkpoint-save cadence
checkpoint-selection rule
```

A0/A1/A2 must receive the same optimization budget.

Do not extend A2 training because the temporal encoder learns slower.

If multiple seeds are affordable, use the same seeds for all variants.

If only one seed is feasible, report:

```text
SINGLE_SEED_EXPLORATORY
```

and do not overclaim robustness.

---

## 17. Frozen Primary Metric

Primary metric:

```text
Dev12 normalized h0 gripper L1
```

Use the repository's already-established implementation/definition.

Direction:

```text
lower is better
```

The exact formula/code path must be recorded.

If this metric cannot be reproduced from the existing evaluation pipeline, stop:

```text
BLOCKED_PRIMARY_METRIC_REPRODUCTION
```

Do not invent a replacement after seeing results.

---

## 18. Frozen Secondary Metrics

Report at minimum:

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

Also report relevant existing action horizons if already part of the project evaluation, but do not promote a secondary horizon to primary after results are known.

---

## 19. Mechanism Diagnostics

For A1 and A2, perform at least the following where technically feasible.

### M1 — Load-zero counterfactual

Run the same Dev observations with the added load representation zeroed / replaced by its normalized baseline.

Report change in predicted gripper action.

Purpose:

> Check whether the model actually uses the load branch.

### M2 — Load perturbation sensitivity

Apply small bounded synthetic perturbations to the load representation offline.

Report:

```text
change in gripper prediction
change in arm prediction
```

The arm prediction should not show disproportionate sensitivity.

### M3 — Episode-level distribution

Report A0/A1/A2 results per Dev episode.

Purpose:

> Detect improvements driven by only a few episodes.

These diagnostics do not replace the primary metric.

---

## 20. Parameter-Count Fairness

Because A2 necessarily has more parameters than A1, report:

```text
extra parameters
percentage parameter increase
latency increase
VRAM increase if measurable
```

If A2 wins, Red-Team Review must consider whether the gain plausibly reflects temporal information or simply increased capacity.

An optional parameter-matched A1 is **not authorized** in r1 because it would add a second architecture variable.

It may be proposed later if needed.

---

## 21. No Contact Gate

A0/A1/A2 must all be:

```text
always-on
```

There is no contact predictor or contact-dependent gating in r1.

Reason:

> Contact gating is reserved for F2.2 if temporal load is first justified.

---

## 22. No Local Residual Policy

Do not add:

```text
delta gripper head
local gripper policy
downstream regulator
high-frequency controller
```

All variants remain within the same OC-ACT policy architecture.

Global-local decomposition is reserved for a later experiment.

---

## 23. No Real-Robot Effectiveness Test

r1 authorizes:

```text
training
offline Dev12 evaluation
mechanism analysis
deployment-cost measurement
```

It does not authorize comparative real-robot effectiveness testing.

A basic load/checkpoint loadability check on the user's RTX 4060 8GB machine is allowed if needed to verify deployability, but it must not be reported as task effectiveness.

---

## 24. Main Outcome Categories

### `TEMPORAL_LOAD_PROMISING`

Use only if:

1. A2 improves over A1 on the frozen primary metric;
2. the gain is directionally consistent across a meaningful fraction of Dev episodes;
3. training health is comparable;
4. A2 does not materially degrade arm-action metrics;
5. load-zero / perturbation diagnostics indicate the temporal branch is actually used;
6. deployment cost remains acceptable.

Interpretation:

> Temporal actuator-load encoding is justified for the next architecture stage.

---

### `SINGLE_FRAME_LOAD_SUFFICIENT`

Use if:

1. A1 improves over A0;
2. A2 does not provide a meaningful or consistent additional gain over A1;
3. A1 mechanism diagnostics indicate the load input is actually used.

Interpretation:

> Keep the simpler single-frame load representation.

---

### `LOAD_HELPFUL_BUT_TEMPORAL_UNRESOLVED`

Use if:

- A1 and/or A2 improve over A0;
- but A2 vs A1 is too small, inconsistent, or unstable to justify a temporal conclusion.

Interpretation:

> Load may help, but the temporal representation question remains unresolved.

---

### `NO_CLEAR_LOAD_BENEFIT`

Use if:

- neither A1 nor A2 shows meaningful stable improvement over A0.

Interpretation:

> Do not automatically proceed to gating or local regulator merely because literature suggests it.

---

### `TEMPORAL_ENCODING_HURTS`

Use if:

- A2 is consistently worse than A1 and/or causes training or prediction instability.

---

### `UNRESOLVED`

Use if:

- baseline semantics differ;
- training budgets differ;
- split integrity fails;
- primary metric cannot be reproduced;
- leakage is found;
- model variants differ beyond the authorized factor.

---

## 25. Completion Criteria

`F2_1_COMPLETE = true` requires:

1. r0 blocked reason preserved in records;
2. A0/A1/A2 base state verified as common 6D;
3. current removed from all three;
4. A1 verified single-frame only;
5. A2 verified causal W15 only;
6. common fusion point verified;
7. same Train36/Dev12 split used;
8. Final12 untouched;
9. matched training budget verified;
10. frozen primary metric reported;
11. per-episode results reported;
12. mechanism diagnostics reported;
13. latency/parameter cost reported;
14. Builder Review evidence complete;
15. Red-Team inputs complete;
16. outcome category assigned.

Completion does not imply load or temporal encoding works.

---

## 26. No Post-Hoc Rescue

After results are known, do not:

- change W15;
- add current;
- add contact gate;
- add FiLM;
- add local residual policy;
- change the fusion point for only one variant;
- add extra training epochs to a weaker variant;
- change the primary metric;
- change seeds selectively;
- use Final12;
- introduce real-robot success as a rescue metric.

Any meaningful change requires a new revision/experiment.

---

## 27. Expected Result Package

Recommended path:

```text
experiments/force/f2_1_load_temporal_encoding_ablation/index1/result/
```

Required:

```text
summary.md
validation.md
r0_blocked_record.md
baseline_identity.json
state_semantics.json
split_manifest.json
variant_configs/
training_manifest.csv
training_health.md
metrics.json
per_episode_metrics.csv
mechanism_review.md
load_zero_ablation.csv
load_perturbation.csv
implementation_diff.md
latency_and_params.json
leakage_audit.md
figures/
proposal.md
```

Recommended figures:

```text
figures/a0_a1_a2_primary_metric.png
figures/per_episode_primary_metric.png
figures/gripper_mae_comparison.png
figures/arm_mae_comparison.png
figures/load_zero_effect.png
figures/load_perturbation_sensitivity.png
figures/training_curves.png
```

---

## 28. Builder Review Questions

Builder Review must verify:

1. Did all variants truly use the same 6D base state?
2. Were `load/current` removed from A0?
3. Was `current` removed from A1/A2?
4. Was A1 strictly single-frame?
5. Was A2 strictly causal W15?
6. Was the fusion point matched?
7. Were training budgets matched?
8. Was the same split used?
9. Was Final12 untouched?
10. Was the primary metric reproduced from the existing pipeline?
11. Were model/checkpoint identities unambiguous?
12. Are deployment costs acceptable?

---

## 29. Red-Team Review Questions

Red-Team Review must ask:

1. Is A2's gain driven by only a few Dev episodes?
2. Is A2 merely learning task phase from load history?
3. Is A2's benefit explained mainly by extra parameters?
4. Does A2 improve gripper error while worsening arm prediction?
5. Does zeroing load materially change predictions?
6. Are perturbation sensitivities physically plausible and bounded?
7. Is A1 already sufficient?
8. Is any conclusion stronger than the evidence warrants?
9. Does this establish temporal encoding only, rather than force control?

---

## 30. Plain-Language Summary

The old F2.1 design was blocked because the current P1 already receives:

```text
load + current
```

inside its normal 8D state.

So this revision rebuilds a clean comparison.

Three models:

```text
A0:
6D position/proprioception only

A1:
same 6D
+ current-frame load

A2:
same 6D
+ last 15 frames of load
```

All three exclude `current`.

The main question is:

> If we already decide to use load, is its recent history actually more useful than one instantaneous value?

A1 vs A0 also tells us whether load itself helps.

The historical 8D P1 is kept only as context and is not allowed to contaminate the main ablation.

---

## 31. Handoff Rule

After execution:

1. stop;
2. return the complete result package;
3. perform Builder Review;
4. perform Red-Team Review;
5. discuss with the user;
6. do not automatically start F2.2;
7. do not add contact gating;
8. do not add local residual policy;
9. do not unseal Final12.

Only if r1 evidence justifies it may a later decision consider:

```text
F2.2:
always-on load representation
vs
contact-gated load representation
```

or another follow-up.
