# decision_F1Q_LOAD_CONFOUND_CONTROL.md

## 0. Experiment Metadata

- **Experiment ID**: `F1Q_LOAD_CONFOUND_CONTROL`
- **Track**: `force`
- **Stage**: `f1q_load_confound_control`
- **Revision**: `r0`
- **Mode**: `DIAGNOSIS`
- **Scientific status**: `EXPLORATORY_ONLY`
- **Parent experiment**: `F1P_CONTACT_SIGNAL_PILOT`
- **Decision owner**: User + Planning Assistant
- **Execution owner**: Codex / implementation agent

Recommended repository location:

```text
experiments/force/f1q_load_confound_control/index1/decision.md
```

This contract does **not** overwrite F1 or F1P. It is a new experiment created because F1P exposed a new scientific uncertainty: whether the observed load gain survives explicit control for position / motion / grasp-phase confounding.

---

## 1. Research Question

F1P showed that adding `load` to the frozen S0 baseline improved W15 episode-balanced BA on the fixed pilot-eval set and satisfied the predeclared `PROMISING_LOAD` exploration rule. However, F1P also showed strong separation between PRECONTACT and CONTACT_SUPPORTED in gripper position / command / motion variables.

The single question for this round is:

> **When PRECONTACT and CONTACT_SUPPORTED samples are explicitly matched or restricted to have similar position / motion / command context, does `load` still provide reproducible incremental contact-state information beyond S0?**

This round is not allowed to answer any broader question such as whether Force improves robot-policy performance, grasp stability, slip detection, or real-robot success rate.

---

## 2. Frozen Hypotheses

### H1 — residual load information exists

After controlling for major S0 confounds, adding `load` still improves contact-state discrimination over the matched/restricted S0 baseline.

### H0 — F1P gain is largely explained by confounding

After controlling for major S0 confounds, the `load` increment becomes small, inconsistent, or unresolved.

Neither H1 nor H0 may be rewritten after seeing results.

---

## 3. Data Scope

### 3.1 Allowed data

Use only the already frozen F1P data:

- **Pilot-fit**: the six previously labeled episodes used by F1P.
- **Pilot-eval**: the six fixed F1P evaluation episodes.
- Use the existing PRECONTACT / CONTACT_SUPPORTED reliable intervals and existing F1P manifests.

### 3.2 Forbidden data

Do not read or use:

- Dev12
- Final12
- any newly collected demonstration
- any real-robot rollout
- any episode outside the already frozen F1P fit/eval subsets

### 3.3 No new labeling by default

No new manual annotation is authorized in this round.

If the existing reliable intervals are insufficient for a valid matched comparison, return `UNRESOLVED` rather than expanding annotation or changing the task.

---

## 4. Frozen Inputs

### 4.1 Base feature family S0

Reuse the exact F1P S0 causal feature construction:

- robot / gripper position channels allowed by F1P
- position differences
- past commands up to `t-1`
- past command differences

Do not add:

- absolute frame index
- normalized episode progress
- episode ID
- future action / future observation
- labels derived from load/current
- any noncausal feature

### 4.2 Load model S1

`S1 = S0 + load`

Use the same causal historical summary family as F1P.

### 4.3 Current

`current` is not part of the primary experiment in this round.

Do not turn this round into a new current investigation. Current may be reported only if trivially available from the frozen F1P pipeline and must remain appendix-only / non-decision-driving.

---

## 5. Confound-Control Construction

The execution must construct a matched/restricted comparison using **S0-observable context only**. Load/current must never be used to form matches.

### 5.1 Primary matching variables

At minimum include the gripper-related variables that produced the strongest F1P phase separation:

- gripper position
- gripper position change / velocity proxy
- previous gripper command
- previous gripper command change

If F1P S0 contains additional arm-motion variables, they may be included only as predeclared secondary matching variables, provided this is done identically for fit and eval and documented before metrics are read.

### 5.2 Matching procedure

Use the following frozen procedure:

1. Standardize matching variables using statistics fit on **Pilot-fit only**.
2. Match each CONTACT_SUPPORTED frame to the nearest PRECONTACT frame **within the same episode** using Euclidean distance in the standardized matching space.
3. Matching is 1:1 without replacement within each episode.
4. A match is accepted only if its standardized distance is `<= 0.75`.
5. Matching must not cross episode boundaries.
6. Matching must not use timestamps, frame index, episode progress, load, current, model prediction, or label confidence beyond the existing reliable-interval inclusion mask.
7. If more than one candidate has identical distance, resolve ties deterministically by the smaller source frame index. Frame index is allowed only for deterministic tie-breaking, never as a matching feature.

### 5.3 Coverage gate

For an episode to count as valid after matching:

- at least **20 matched pairs** must remain, and
- each class must retain at least 20 frames after matching by construction.

For the experiment to be evaluable:

- at least **4/6 Pilot-fit episodes** must be valid, and
- at least **4/6 Pilot-eval episodes** must be valid.

If this gate fails, the scientific outcome is `UNRESOLVED_MATCHING_COVERAGE`.

Do not relax the distance threshold after seeing coverage.

---

## 6. Balance / Confound Audit

Before reading probe performance, generate a balance audit for the matched data.

For each matching variable, report:

- class mean and standard deviation before matching
- class mean and standard deviation after matching
- standardized mean difference (SMD) before matching
- SMD after matching

Primary balance requirement:

> After matching, every primary matching variable must have `|SMD| <= 0.20` on the pooled Pilot-eval matched set.

Also report the same metrics per episode when sample size permits.

If coverage passes but the pooled eval balance requirement fails, outcome is `UNRESOLVED_RESIDUAL_CONFOUND`.

No probe result may upgrade this status.

---

## 7. Probe Model

Reuse the F1P fixed probe family and do not tune it.

### 7.1 Main window

- **Primary window**: `W15`

W1/W9 may be reported as appendix sensitivity checks if already supported by the pipeline, but they are not allowed to drive the round conclusion.

### 7.2 Classifier

Fixed logistic regression:

- penalty: L2
- `C = 1`
- class weighting: balanced
- solver: `lbfgs`
- `max_iter = 2000`
- `tol = 1e-4`
- seed: `20261015`
- decision threshold: `0.5`

### 7.3 Scaling

- Fit scaler on matched Pilot-fit only.
- Apply the frozen scaler to matched Pilot-eval.
- No whole-dataset scaler.
- No eval-aware normalization.

### 7.4 Models

Primary comparison only:

- `M0_matched`: matched S0
- `M1_matched`: matched S0 + load

No hyperparameter sweep, feature selection, alternate classifier, neural network, threshold optimization, or episode-specific tuning.

---

## 8. Primary Metric and Statistical Summary

### 8.1 Primary metric

For each valid Pilot-eval episode compute:

```text
Delta_i = BA(M1_matched)_i - BA(M0_matched)_i
```

Primary summary:

```text
mean_episode_delta = equal-weight mean of Delta_i over valid eval episodes
```

Also report:

- BA for M0 and M1 per episode
- Macro F1
- PRECONTACT recall
- CONTACT_SUPPORTED recall
- confusion matrix
- number of matched pairs
- matching-distance distribution

### 8.2 Exploratory uncertainty

Use paired bootstrap over valid evaluation episodes:

- 2000 bootstrap samples
- seed `20261015`
- 95% percentile interval

This interval is exploratory only and must not be described as confirmatory inference.

---

## 9. Frozen Outcome Rules

Outcome evaluation order is fixed.

### 9.1 `UNRESOLVED_MATCHING_COVERAGE`

Use if coverage gate in Section 5.3 fails.

### 9.2 `UNRESOLVED_RESIDUAL_CONFOUND`

Use if coverage passes but the pooled Pilot-eval post-match balance requirement `|SMD| <= 0.20` fails for any primary matching variable.

### 9.3 `RESIDUAL_LOAD_SIGNAL_PROMISING`

Only if all engineering, coverage, leakage, and balance checks pass, and:

1. `mean_episode_delta >= +0.03` BA, and
2. at least `ceil(2N/3)` valid eval episodes have positive `Delta_i`, where N is the number of valid eval episodes.

This means only:

> load retains a meaningful exploratory increment after explicit control for major measured position / motion confounds.

It does **not** mean Force policy efficacy is proven.

### 9.4 `SMALL_RESIDUAL_LOAD_SIGNAL`

If checks pass and:

```text
0 < mean_episode_delta < 0.03
```

or fewer than `ceil(2N/3)` episodes are positive.

Interpretation:

> some residual information may remain, but evidence is not strong enough to justify claiming a robust contact-specific signal from this pilot.

### 9.5 `NO_CLEAR_RESIDUAL_GAIN`

If checks pass and:

```text
mean_episode_delta <= 0
```

Interpretation:

> after measured confound control, this pilot does not show a clear load increment.

This does not prove load is universally useless.

---

## 10. Leakage and Validity Requirements

Execution must explicitly verify and report:

- fit/eval episode separation preserved
- no Dev/Final access
- no cross-episode matching
- no load/current used in matching
- no future observation/action leakage
- commands restricted to `<= t-1`
- scaler fit only on matched Pilot-fit
- no label generation from load/current
- no post-result threshold or distance-threshold change
- no discarded negative episodes after metrics are known

Any violation that can affect the scientific conclusion makes the round `INVALID`.

---

## 11. Required Evidence / Deliverables

Under:

```text
experiments/force/f1q_load_confound_control/index1/result/
```

provide at minimum:

```text
summary.md
validation.md
subset_manifest.json
matching_config.json
matching_coverage.csv
balance_audit.csv
per_episode_metrics.csv
metrics.json
bootstrap.json
leakage_audit.md
confound_review.md
proposal.md
figures/
```

### Required figures

At minimum:

1. before-vs-after matching distribution for gripper position
2. before-vs-after matching distribution for gripper motion / delta
3. before-vs-after matching distribution for previous gripper command
4. per-episode `BA(S1)-BA(S0)` after matching
5. matched-pair distance / coverage summary

Raw run artifacts should remain under the standard raw-results location used by the project workflow.

Update:

- `workflow/STATUS.md`
- project work log

Do not alter historical F1/F1P results.

---

## 12. Permissions / Prohibitions

### Authorized

- offline diagnosis using the frozen F1P subsets
- deterministic matching / restriction described in this contract
- fixed logistic probes
- generation of diagnostic evidence and proposal

### Not authorized

- robot policy training
- fine-tuning P1 / OC-ACT
- adding Force head
- adding auxiliary loss to policy
- changing P1 architecture
- automatic gripper tightening
- closed-loop force control
- real-robot testing
- new data collection
- new manual annotation
- reading Dev12
- reading Final12
- changing the frozen matching threshold after looking at results
- changing the primary metric / success rule
- automatically starting F2
- pushing repository changes unless separately authorized by the user/project rules

---

## 13. Completion Definition

### `F1Q_COMPLETE = true`

Only if:

- required artifacts are delivered,
- matching/coverage/balance/leakage checks are reported,
- fixed probes complete on valid matched data,
- outcome is assigned using Section 9 exactly,
- STATUS and work log are updated.

`F1Q_COMPLETE = true` means the experiment contract was executed completely.

It does **not** mean H1 is true.

Possible scientific outcomes remain:

- `RESIDUAL_LOAD_SIGNAL_PROMISING`
- `SMALL_RESIDUAL_LOAD_SIGNAL`
- `NO_CLEAR_RESIDUAL_GAIN`
- `UNRESOLVED_MATCHING_COVERAGE`
- `UNRESOLVED_RESIDUAL_CONFOUND`
- `INVALID`

---

## 14. Stop Rule / Handoff

After producing the result package:

1. stop execution;
2. do not start policy training or F2;
3. hand the result package back to the user + Planning Assistant;
4. next step must go through Builder Review and Red-Team Review;
5. only after discussion may a new decision authorize a minimal Force/contact-aware policy experiment.

---

## 15. Scientific Interpretation Boundary

The strongest statement this experiment may support is:

> On the frozen F1P pilot subsets, `load` retains an exploratory incremental ability to distinguish PRECONTACT from CONTACT_SUPPORTED after explicit matching on major measured gripper position / motion / command confounds.

It may **not** support claims that:

- `load` is a calibrated force sensor;
- the system measures Newton-level grasp force;
- slip is detected;
- stable grasp is inferred;
- Force policy improves manipulation success;
- the effect generalizes to Dev/Final or new object placements;
- transparent-object manipulation has been solved.

