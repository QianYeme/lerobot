# decision_F2_1_LOAD_TEMPORAL_ENCODING_ABLATION.md

## 0. Contract Header

- Experiment ID: `F2_1_LOAD_TEMPORAL_ENCODING_ABLATION`
- Revision: `r0`
- Mode: `ABLATION / FEASIBILITY`
- Track: `force`
- Status at creation: `FROZEN / NOT_RUN`
- Scientific scope: `EXPLORATORY_ARCHITECTURE_ABLATION`
- Parent evidence:
  - `F1P_CONTACT_SIGNAL_PILOT`: `PROMISING_LOAD`
  - `F1Q_LOAD_CONFOUND_CONTROL`: `UNRESOLVED_MATCHING_COVERAGE`
  - `F1RM_TELEOP_LOAD_RETROSPECTIVE_MATCH`: `UNRESOLVED_RESIDUAL_CONFOUND`
- Baseline:
  - current P1 / OC-ACT
- External design motivation:
  - recent force-aware manipulation work suggests temporal/multimodal force encoding can be more useful than naive single-frame fusion
  - this round tests only that factor in the current project

Recommended repository path:

```text
experiments/force/f2_1_load_temporal_encoding_ablation/index1/decision.md
```

This is a new experiment contract.
It does not overwrite previous F1/F2 drafts.

---

## 1. Single Research Question

This round answers only:

> Does a short temporal history of `gripper.load` provide more useful policy information than a single current `load` value when added to the existing P1 / OC-ACT policy?

The round is not allowed to answer:
- whether contact gating helps;
- whether a local residual gripper policy helps;
- whether `current` helps;
- whether higher-frequency local control helps;
- whether force-aware control is finally effective.

Those remain future questions.

---

## 2. Frozen Variants

Exactly three primary variants are allowed.

### A0 — Baseline

```text
P1 / OC-ACT
```

No new load branch.

### A1 — Single-Frame Load

```text
P1 / OC-ACT
+
current-frame gripper.load
```

The load signal is incorporated using the smallest architecture-compatible fusion that preserves the P1 backbone semantics.

### A2 — Temporal Load

```text
P1 / OC-ACT
+
W15 gripper.load history
+
small temporal encoder
```

The temporal encoder may summarize the 15-frame causal history into one load feature/token.

No future load samples are allowed.

---

## 3. Frozen Scientific Contrast

Primary contrast:

```text
A2 vs A1
```

This answers whether temporal information is useful beyond a single instantaneous actuator-load value.

Secondary contrasts:

```text
A1 vs A0
A2 vs A0
```

These answer whether either load-aware variant adds value relative to P1.

The experiment must not be reframed post-hoc to focus only on whichever pair looks best.

---

## 4. Baseline Identity

Before training:

Record and freeze:

```text
P1 checkpoint identity
git commit / HEAD
training config
dataset
Train/Dev split
camera inputs
state fields
action dimension
chunk length
temporal ensembling settings
normalization
training budget
```

If the baseline identity cannot be verified, stop:

```text
BLOCKED_BASELINE_IDENTITY
```

Do not silently replace P1 with another checkpoint.

---

## 5. Dataset and Split

Use the existing formal dataset and frozen development split already defined by the project.

Expected:

```text
Train36
Dev12
Final12 sealed
```

Rules:

- Train36 may be used for training.
- Dev12 may be used for model selection/evaluation in this ablation if already authorized under the project workflow.
- Final12 remains sealed.
- Do not resample or redraw the split.
- Do not frame-random-split adjacent frames across Train/Dev.
- No new robot data collection is authorized.

If repository status indicates a more recent frozen split contract, the executor must stop and report the mismatch rather than guessing.

---

## 6. Input Semantics

### A0

Use the existing P1 inputs exactly.

### A1

Add only:

```text
load_t
```

No `current`.

No future signal.

### A2

Add:

```text
load_(t-14:t)
```

using a causal W15 history.

For startup frames with fewer than 15 available frames:

```text
pad causally or use the available prefix
```

The chosen startup handling must be frozen and identical across training/evaluation.

---

## 7. Temporal Encoder

A2 must use a small encoder.

Allowed examples:

- shallow MLP over normalized W15 vector;
- 1D convolution with very small capacity;
- small GRU;
- small temporal Transformer.

Preferred r0 implementation:

```text
small MLP or 1D temporal encoder
```

The encoder should be lightweight enough for deployment on the user's RTX 4060 8GB laptop.

Do not introduce a large backbone.

The exact architecture must be documented before training.

---

## 8. Fusion Rule

This round tests temporal history, not complex multimodal fusion.

Therefore the load feature must enter P1 using one simple, fixed fusion mechanism.

Allowed examples:

```text
concatenate with robot-state token
or
project to one additional token and append to the policy token sequence
```

The same fusion position must be used for A1 and A2 wherever technically possible.

Only the load representation differs:

```text
A1 = single-frame representation
A2 = temporal representation
```

Do not give A2 a richer downstream fusion path than A1.

---

## 9. Frozen Architecture Constraints

Do not add:

- contact predictor;
- contact gate;
- local residual policy;
- current branch;
- tactile branch;
- force head;
- target-force output;
- high-frequency auxiliary controller;
- FiLM specific to load;
- extra vision backbone;
- new object detector;
- new action space.

The only architecture difference under study is:

```text
no load
vs
single-frame load
vs
W15 load history
```

---

## 10. Training Initialization

Preferred fairness rule:

- A0 uses the verified P1 baseline checkpoint or retraining protocol as the baseline reference.
- A1 and A2 must share the same initialization strategy.

If load-aware variants require new modules:

```text
P1 shared weights initialized identically
new load-specific module initialized using the same seed policy
```

Do not allow one variant to continue training substantially longer because it underperforms.

---

## 11. Training Budget

Use the same frozen training budget for A1 and A2.

At minimum freeze:

```text
number of optimizer steps / epochs
batch size
learning rate schedule
optimizer
seed(s)
checkpoint selection rule
```

Do not increase A2 training simply because its encoder is temporal.

If compute allows, use the same small set of seeds for A1 and A2.

If only one seed is affordable, state explicitly that evidence is preliminary.

---

## 12. Main Offline Metrics

Use the project’s existing policy-level offline evaluation metrics where valid.

At minimum report:

1. action error on Dev12;
2. arm action error;
3. gripper action error;
4. temporal consistency / smoothness if already supported;
5. phase-specific gripper behavior if derivable without new post-hoc labels;
6. compute cost / latency;
7. parameter increase.

Do not invent a new primary metric after results are seen.

---

## 13. Primary Metric

Primary metric for this ablation:

> Dev12 gripper-action quality under the same frozen evaluation protocol already used for P1-related comparisons.

If the repository has an already-established scalar gripper metric, use it.

If no single established metric exists, the executor must report:

```text
BLOCKED_PRIMARY_METRIC_DEFINITION
```

and provide a proposal rather than inventing one after training.

---

## 14. Secondary Mechanism Analysis

To understand whether temporal load is actually being used:

For A1 and A2, report where feasible:

- sensitivity of predicted gripper action to perturbing load input;
- magnitude of learned load-feature activations;
- comparison of predictions with load branch zeroed;
- behavior around grasp/contact-related portions of Dev trajectories.

These are mechanism diagnostics.

They are not allowed to redefine the main pass/fail result.

---

## 15. Real-Robot Boundary

No real-robot regulated closed-loop experiment is authorized in this round by default.

This round is primarily:

```text
training + offline evaluation + mechanism analysis
```

If an existing project convention already authorizes a tiny deployment equivalence check, that may be used only to verify the model runs.

No comparative real-robot effectiveness claim may be made without a later decision.

---

## 16. Main Outcome Categories

### `TEMPORAL_LOAD_PROMISING`

Use only if all are true:

1. A2 outperforms A1 on the frozen primary metric;
2. the improvement is not explained by obvious training instability;
3. A2 is not clearly worse than A0 on arm-action behavior;
4. compute/deployment cost remains acceptable;
5. mechanism diagnostics suggest A2 uses temporal load information rather than ignoring it.

Interpretation:

> W15 temporal actuator-load encoding is worth carrying into the next architecture stage.

---

### `SINGLE_FRAME_SUFFICIENT`

Use if:

- A1 improves over A0;
- A2 does not provide a meaningful or stable additional gain over A1.

Interpretation:

> Temporal encoding is not justified by current evidence; use the simpler single-frame load representation in the next stage.

---

### `NO_CLEAR_LOAD_BENEFIT`

Use if:

- neither A1 nor A2 provides a meaningful or stable gain over A0 under the frozen evaluation.

Interpretation:

> Do not automatically proceed to more complex contact gating or local policies merely because they are fashionable.

---

### `TEMPORAL_ENCODING_HURTS`

Use if:

- A2 is consistently worse than A1 and/or destabilizes gripper prediction.

Interpretation:

> W15 history is not supported in its current form.

---

### `UNRESOLVED`

Use if:

- training is unhealthy;
- baseline identity is uncertain;
- metric definition is invalid;
- checkpoints are incomparable;
- implementation differs beyond the intended variable.

---

## 17. Completion Criteria

`F2_1_COMPLETE = true` requires:

1. A0 baseline identity verified;
2. A1 and A2 implemented under matched conditions;
3. identical dataset/split used;
4. identical training budget used;
5. Final12 untouched;
6. required metrics reported;
7. compute/latency reported;
8. implementation-difference audit complete;
9. mechanism diagnostics complete where technically feasible;
10. outcome category assigned;
11. result package complete.

Completion does not imply that temporal load works.

---

## 18. No Post-Hoc Rescue

After seeing results, do not:

- change W15 to another window;
- add current;
- add a contact gate;
- switch fusion mechanism only for A2;
- give A2 more training;
- change the checkpoint-selection rule;
- redefine the primary metric;
- unseal Final12;
- add real-robot success as a rescue metric.

Any scientifically meaningful change requires a new revision or experiment.

---

## 19. Expected Deliverables

Recommended result path:

```text
experiments/force/f2_1_load_temporal_encoding_ablation/index1/result/
```

Minimum:

```text
summary.md
validation.md
baseline_identity.json
variant_configs/
training_manifest.csv
training_health.md
metrics.json
per_variant_metrics.csv
mechanism_review.md
implementation_diff.md
latency_and_params.json
leakage_audit.md
figures/
proposal.md
```

Recommended figures:

```text
figures/a0_a1_a2_primary_metric.png
figures/gripper_error_comparison.png
figures/load_sensitivity.png
figures/temporal_feature_activity.png
figures/training_curves.png
```

---

## 20. Builder Review Questions

Builder Review must answer:

1. Were A0/A1/A2 identical except for load representation?
2. Was A1 truly single-frame only?
3. Was A2 truly causal W15 only?
4. Was current excluded?
5. Was contact gating excluded?
6. Was local residual control excluded?
7. Were training budgets identical?
8. Was Final12 untouched?
9. Were arm-policy components unintentionally changed?
10. Are latency/parameter costs acceptable for the RTX 4060 8GB deployment target?

---

## 21. Red-Team Review Questions

Red-Team Review must ask:

1. Is any A2 gain simply due to more parameters?
2. Is temporal history leaking implicit episode progress?
3. Does A2 improve gripper prediction while harming arm behavior?
4. Are improvements concentrated in only a few Dev episodes?
5. Are gains large enough to justify added complexity?
6. Could load simply be serving as a phase cue?
7. Does zeroing the load feature meaningfully change A2 outputs?
8. Is the correct conclusion about temporal representation, not about proven force control?

---

## 22. Plain-Language Summary

This round compares only three versions:

```text
A0: original P1

A1: P1 + current load value

A2: P1 + last 15 frames of load
```

The purpose is not to build the final force-control system yet.

It only asks:

> Is one instant of actuator load enough, or does the recent load history contain more useful information for gripper behavior?

If A2 is clearly better than A1, we carry temporal encoding into the next stage.

If A1 is already enough, we keep the simpler design.

If neither helps, we stop before adding more complicated contact gates or local policies.

---

## 23. Handoff Rule

After execution:

1. stop;
2. return the full result package;
3. perform Builder Review;
4. perform Red-Team Review;
5. discuss with the user;
6. do not automatically start F2.2;
7. do not add contact gating or local residual policy without a new decision.

Possible next step only if justified:

```text
F2.2:
always-on load
vs
contact-gated load
```

That future round is not authorized here.
