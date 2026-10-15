# decision_F1R_ACTUATOR_LOAD_CONTACT_INTERVENTION.md

## 0. Contract Header

- Experiment ID: `F1R_ACTUATOR_LOAD_CONTACT_INTERVENTION`
- Revision: `r0`
- Mode: `DIAGNOSIS`
- Track: `force`
- Status at creation: `FROZEN / NOT_RUN`
- Scientific scope: `EXPLORATORY_SENSOR_CHARACTERIZATION`
- Parent evidence:
  - `F1P_CONTACT_SIGNAL_PILOT`: `PROMISING_LOAD`
  - `F1Q_LOAD_CONFOUND_CONTROL`: `UNRESOLVED_MATCHING_COVERAGE`

Recommended repository path:

```text
experiments/force/f1r_actuator_load_contact_intervention/index1/decision.md
```

This decision is a new experiment contract. It does not overwrite F1, F1P, or F1Q.

---

## 1. Single Research Question

This round answers only one primary question:

> Under nearly identical gripper command, gripper position, closing motion, and fixed arm pose, does `gripper.load` show a repeatable response specifically associated with physical cup contact?

This round does **not** test whether a force-aware policy improves robot task success.

---

## 2. Motivation

F1P found that adding `load` improved PRECONTACT vs CONTACT_SUPPORTED decoding under the frozen pilot protocol.

F1Q attempted to control gripper position / motion / previous-command confounds by matching naturally occurring demonstration frames, but the existing demonstrations did not contain enough within-episode counterfactual pairs. Therefore F1Q ended as:

```text
UNRESOLVED_MATCHING_COVERAGE
```

The next step is not to loosen matching thresholds or retrain a policy. Instead, this experiment actively creates controlled counterfactual conditions:

```text
same / near-same arm pose
same gripper closing program
same nominal motion
different physical condition:
    FREE_CLOSE vs CUP_CONTACT
```

This allows direct characterization of actuator feedback.

---

## 3. Hypothesis

Primary exploratory hypothesis:

> `gripper.load` contains a repeatable contact-related response that remains distinguishable between FREE_CLOSE and CUP_CONTACT under controlled motion.

Secondary exploratory question:

> Does `gripper.current` provide any additional stable contact-related response under the same controlled intervention?

The round must not call either signal "true force" or report Newton-level force because no physical force calibration is assumed.

---

## 4. Hardware and Object

Use the same robot / gripper and the same half-full transparent plastic cup used in the current project.

Approximate cup geometry currently known:

- bottom diameter: ~50 mm
- top diameter: ~60 mm

Do not change gripper hardware during this round.

Do not intentionally deform, crush, or damage the cup.

---

## 5. Core Experimental Principle

The arm and wrist remain fixed during each closing trial.

Only the gripper executes the closing motion.

Two primary experimental conditions:

### Condition A — FREE_CLOSE

No cup or other object is present between the fingers.

The gripper executes the frozen closing program.

### Condition B — CUP_CONTACT

The cup is placed between the fingers.

The arm pose and gripper closing program are the same as in FREE_CLOSE.

The cup must be positioned so that the gripper fingers establish contact during closing without requiring arm movement.

The main contrast is:

```text
FREE_CLOSE
vs
CUP_CONTACT
```

---

## 6. Minimal Pilot Design

The initial pilot uses one fixed arm pose and one fixed closing program.

Required trials:

```text
FREE_CLOSE:   5 valid trials
CUP_CONTACT:  5 valid trials
```

Total required valid trials:

```text
10
```

This is intentionally small.

Do not automatically expand to multiple speeds, multiple arm poses, or dozens of repetitions in the same run.

Expansion requires a new decision or explicit revision after review.

---

## 7. Frozen Gripper Closing Program

Before collecting the 10 formal trials, the executor may perform a short engineering setup to determine a safe closing range.

The formal run must then freeze:

1. initial gripper command / opening;
2. final gripper command / opening;
3. command trajectory or step sequence;
4. command update rate;
5. trial duration;
6. sampling frequency;
7. maximum safe closure.

The same frozen closing program must be used for all 10 formal trials.

The executor must record the actual values in:

```text
result/protocol.json
```

### Safety constraint

The final closing command must be selected conservatively so that the CUP_CONTACT condition does not intentionally crush or severely deform the cup.

If an automatic safety stop or operator emergency stop is triggered, that trial is invalid and must be reported.

---

## 8. Cup Placement

For the 5 CUP_CONTACT trials:

- use one frozen nominal cup placement;
- cup orientation should remain the same;
- cup fill state should remain the same;
- do not intentionally reposition the arm between trials;
- minor unavoidable manual placement variation is allowed and must not be hidden.

The executor should mark or otherwise reproduce the cup placement as consistently as practical.

This round does not claim spatial generalization.

---

## 9. Required Signals

For every frame / sample, save at least:

```text
timestamp or monotonic sample index
episode / trial ID
condition: FREE_CLOSE or CUP_CONTACT
gripper.pos
gripper.load
gripper.current or gripper.curr
commanded gripper target
```

If already available without changing the experiment semantics, also retain:

```text
full robot state
full action
arm joint positions
camera frame timestamp
```

Raw signals must not be discarded after preprocessing.

---

## 10. Video Requirement

Record synchronized or time-alignable RGB video for every formal trial if the existing collection pipeline supports it.

The purpose of video is diagnostic only:

- identify the approximate first physical finger–cup contact;
- verify the cup was actually present / absent;
- verify the arm stayed stationary;
- identify accidental cup motion, severe deformation, or failed trials.

Video is not a policy-training dataset in this round.

---

## 11. Contact Annotation

For each CUP_CONTACT trial, identify a conservative contact interval or contact onset using RGB evidence.

Rules:

- do not derive contact labels from `load` or `current`;
- uncertain onset frames must be marked uncertain rather than forced;
- the operator does not need to click every frame;
- interval endpoints are sufficient if tooling expands them programmatically.

FREE_CLOSE trials are labeled as no-object-contact by experimental construction, subject to video validation.

---

## 12. Valid Trial Criteria

A FREE_CLOSE trial is valid only if:

1. no object is between the fingers;
2. the arm remains stationary apart from negligible mechanical noise;
3. the full frozen closing program is executed;
4. all required signals are recorded;
5. there is no emergency stop or corrupted log.

A CUP_CONTACT trial is valid only if:

1. the cup is present in the frozen nominal location;
2. physical finger–cup contact is visible or otherwise independently verifiable;
3. the arm remains stationary;
4. the full safe closing program is executed unless a predefined safety stop terminates it;
5. all required signals are recorded;
6. no severe unintended cup damage occurs.

Invalid trials may be repeated only to reach 5 valid trials per condition.

All invalid attempts must still be listed in the result log.

---

## 13. Primary Analysis

The analysis must compare FREE_CLOSE and CUP_CONTACT under the frozen closing program.

At minimum, report:

### A. Raw trajectory plots

For every trial:

```text
gripper command vs time
gripper.pos vs time
gripper.load vs time
gripper.current vs time
```

Contact onset / interval should be overlaid for CUP_CONTACT where available.

### B. Position-aligned comparison

Compare FREE_CLOSE and CUP_CONTACT at similar `gripper.pos`.

This is important because simple time alignment alone can confound contact with actuator lag.

### C. Command-aligned comparison

Compare both conditions at the same or nearest commanded gripper target.

### D. Contact-relative traces

For CUP_CONTACT trials, align traces around independently annotated contact onset where possible:

```text
t = 0 = contact onset
```

Show at least a short pre-contact and post-contact window.

---

## 14. Primary Metrics

Primary signal:

```text
gripper.load
```

Secondary signal:

```text
gripper.current
```

For load, report at least:

1. pre-contact baseline mean/std;
2. post-contact mean/std;
3. change from local pre-contact baseline;
4. matched-position FREE_CLOSE vs CUP_CONTACT difference;
5. per-trial effect direction;
6. across-trial mean and variability;
7. simple effect-size summary;
8. overlap / separability visualization.

No mandatory classifier is required for completion.

A small fixed diagnostic classifier may be included only as secondary evidence if it does not replace the direct signal analysis.

---

## 15. Exploratory Decision Categories

The following categories are descriptive gates, not final scientific proof.

### `CONTACT_RESPONSE_PROMISING`

Use only if all are true:

1. at least 4/5 CUP_CONTACT trials show the same-direction load response after independently identified contact;
2. matched-position / matched-command comparisons show a consistent load difference from FREE_CLOSE;
3. the difference is not explained by obvious logging failure, arm movement, or cup-placement artifact;
4. the response magnitude is visibly larger than normal FREE_CLOSE trial-to-trial variation.

Interpretation:

> `load` is a promising actuator-feedback signal for contact-aware modeling and may justify a minimal reactive regulator experiment.

### `WEAK_OR_VARIABLE_RESPONSE`

Use if:

- some contact-related load change is visible,
- but repeatability is poor, overlap is large, or effect direction is inconsistent.

Interpretation:

> The signal may contain useful information but is not yet stable enough to justify strong contact-control claims.

### `NO_CLEAR_CONTACT_RESPONSE`

Use if:

- FREE_CLOSE and CUP_CONTACT traces substantially overlap under the controlled protocol,
- and no repeatable post-contact load response is observed.

Interpretation:

> The current `load` signal is not supported as a primary contact cue under this protocol.

This does **not** prove the signal is useless in every manipulation setting.

### `UNRESOLVED`

Use if:

- contact cannot be independently verified;
- closing programs differ materially between conditions;
- too few valid trials remain;
- logging is incomplete;
- safety limitations prevent meaningful contact;
- protocol equivalence is not maintained.

---

## 16. Completion Criteria

`F1R_COMPLETE = true` requires:

1. 5 valid FREE_CLOSE trials;
2. 5 valid CUP_CONTACT trials;
3. frozen closing protocol documented;
4. required raw signals retained;
5. independent contact verification for CUP_CONTACT;
6. required plots and aligned comparisons generated;
7. trial validity table complete;
8. one scientific outcome category assigned;
9. Builder-facing summary and evidence package complete.

Important:

```text
F1R_COMPLETE = true
```

means the experiment was completed correctly.

It does **not** mean:

```text
CONTACT_RESPONSE_PROMISING
```

must be true.

---

## 17. Forbidden Changes

This round does not authorize:

- training P1 / OC-ACT;
- adding a force head;
- adding a contact auxiliary loss;
- changing the deployed policy;
- closed-loop force control;
- automatic gripper tightening based on load;
- target-force control;
- reading Dev12;
- reading Final12;
- modifying Train/Dev/Final splits;
- collecting a new policy demonstration dataset;
- claiming `load` is calibrated force;
- reporting force in Newtons without an independent calibration experiment;
- automatically expanding to multiple speeds / placements after seeing results;
- silently changing the frozen closing protocol mid-run.

Any scientifically meaningful protocol change requires a new revision or a new experiment.

---

## 18. Result Package

Recommended path:

```text
experiments/force/f1r_actuator_load_contact_intervention/index1/result/
```

Minimum deliverables:

```text
summary.md
validation.md
protocol.json
trial_manifest.csv
trial_validity.csv
contact_annotations.csv
metrics.json
per_trial_metrics.csv
raw/
figures/
proposal.md
```

Recommended figures:

```text
figures/load_time_all_trials.png
figures/current_time_all_trials.png
figures/load_vs_gripper_position.png
figures/current_vs_gripper_position.png
figures/contact_aligned_load.png
figures/free_vs_contact_matched_position.png
```

`proposal.md` may suggest next steps but must not execute them.

---

## 19. Human Task — What the User Actually Needs to Do

The human operator's required work is intentionally minimal.

### Before formal collection

1. place the robot in one safe fixed pose;
2. confirm the arm will remain stationary;
3. establish one safe gripper closing program;
4. confirm logging and video work;
5. mark a repeatable cup placement.

### Formal collection

Collect:

```text
5 × FREE_CLOSE
5 × CUP_CONTACT
```

For FREE_CLOSE:
- leave the gripper empty;
- run the frozen closing program.

For CUP_CONTACT:
- place the same half-full cup at the marked location;
- run the exact same closing program.

The human does not need to interpret the load signal during collection.

The human should stop immediately if:
- the cup is being dangerously crushed;
- the gripper behaves unexpectedly;
- the arm moves unexpectedly;
- hardware safety is at risk.

---

## 20. Handoff Rule

After execution, stop.

Do not automatically start a force-aware policy experiment.

The result package must first undergo:

1. Builder Review;
2. Red-Team Review;
3. discussion with the user;
4. explicit next-step approval.

Only after that may a later decision consider:

```text
P1 / OC-ACT
+
reactive contact/load regulator
```

or another force-aware architecture.

---

## 21. Plain-Language Summary

This experiment is not teaching the robot to use force.

It is checking whether the robot's existing `gripper.load` field behaves like a useful contact signal.

The test deliberately runs the same gripper-closing motion:

```text
once with no cup
once with the cup
```

while keeping the arm still.

If the load curve changes reliably only when the fingers contact the cup, then later force/contact-aware modeling has a stronger scientific foundation.

If it does not, the project should avoid building a large force-control branch around an unreliable signal.
