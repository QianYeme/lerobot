# decision_R4_INFERENCE_REPRODUCIBILITY_AND_FINAL_DATASET_GATE.md

## 0. Contract Header

- Experiment ID: `R4_INFERENCE_REPRODUCIBILITY_AND_FINAL_DATASET_GATE`
- Revision: `r0`
- Mode: `OFFLINE_DEBUG + NEXT_STAGE_GATE`
- Robot execution: `NO`
- New training: `NO`
- New architecture: `NO`
- New dataset collection: `NO` in this decision
- Primary blocker:
  - same checkpoint + nominally same evaluation produced max difference ≈ `0.117225647`
  - frozen reproducibility tolerance was `< 0.001`
- Purpose:
  1. locate the source of inference non-reproducibility;
  2. determine whether the current multi-seed conclusions are trustworthy;
  3. only after this passes, authorize final-dataset collection and later seed-scaling experiments.

Recommended repository path:

```text
experiments/force/r4_inference_reproducibility/index1/decision.md
```

---

# 1. Why This Is the Immediate Next Step

Current evidence already suggests:

```text
Global OC-ACT:
- strong seed sensitivity
- mixed Full60 spatial gains

Local MLP:
- comparatively stable
```

However, the current evaluation pipeline has an unresolved issue:

```text
same checkpoint
same intended evaluation
fresh inference twice
→ max difference ≈ 0.117225647
```

Before studying training-seed variance further, evaluation variance must be separated from training variance.

This decision changes no scientific model variable.

---

# 2. Core Research Question

> Is the 0.117 difference caused by model/kernel nondeterminism, process/runtime state, preprocessing/data loading, temporal state/cache, or evaluation implementation?

The experiment must localize the first layer at which divergence appears.

---

# 3. Frozen Model for Audit

Primary checkpoint:

```text
G-Load-60 seed3 / 80k
```

Secondary checkpoint:

```text
G-Load-60 seed1 / 80k
```

Reason:
- seed3 is the core paired Full60 model;
- seed1 provides the second-seed repeat audit required by the previous contract.

No retraining.

---

# 4. Audit Hierarchy

Run in strict order.

## A. Same tensor, same process, same loaded model

1. Load one frozen checkpoint once.
2. Construct one frozen batch/tensor input once.
3. Save exact input tensors.
4. Set model to eval mode.
5. Disable augmentation.
6. Reset any temporal/cache state.
7. Run forward pass twice in the same Python process.

Compare:

```text
raw model outputs
arm chunk
gripper chunk
intermediate feature if easy to expose
```

If max difference > 0.001:

```text
FAIL_A_MODEL_OR_KERNEL_NONDETERMINISM
```

Stop interpretation and inspect:
- dropout/eval state;
- stochastic layers;
- CUDA nondeterministic operations;
- mixed precision;
- TF32;
- temporal state mutation.

---

## B. Same saved tensor, separate fresh processes

Only if A passes.

1. Save the exact frozen input tensor batch to disk.
2. Launch two independent Python processes.
3. Load the same checkpoint/config.
4. Load the exact same tensor file.
5. Run one inference each.

If A passes but B fails:

```text
FAIL_B_PROCESS_RUNTIME_NONDETERMINISM
```

Inspect:
- environment flags;
- AMP/autocast;
- CUDA/cuDNN deterministic settings;
- random seeds;
- library/runtime initialization.

---

## C. Same sample IDs, full preprocessing twice

Only if A and B pass.

Run the complete evaluation pipeline twice from raw dataset records, using the same fixed sample IDs/order.

Compare in order:

```text
raw sample IDs
decoded image tensors
normalized images
robot state tensors
load tensors
object-centric features
model inputs
model outputs
```

Record the first stage where numerical divergence appears.

If input tensors differ:

```text
FAIL_C_PREPROCESS_OR_DATALOADER
```

Inspect:
- random augmentation;
- image transforms;
- DataLoader worker seeds;
- sample order;
- normalization;
- cached feature generation.

---

## D. Full fresh replay twice

Only if A–C pass.

Run the complete Full60 replay twice.

Requirements:

```text
same sample order
same checkpoint
same config
same preprocessing
same reset policy
```

Compare:
- every sample prediction;
- every aggregate metric;
- max absolute output difference.

Pass condition:

```text
max absolute action difference < 0.001
```

---

# 5. Temporal State / Cache Audit

Because ACT-style execution may use temporal ensembling or cached predictions, explicitly verify whether evaluation is intended to be stateless.

For every sample/batch:
- reset temporal ensembling state if replay is per-observation independent;
- or preserve an identical deterministic episode sequence if replay is sequential.

Log:

```text
episode boundary reset
temporal buffer state
previous-action cache
observation history
```

A replay that starts from different hidden/cache state is not considered the same inference.

---

# 6. Determinism Configuration Audit

Record, do not silently modify at first:

```text
Python seed
NumPy seed
torch seed
torch.cuda seed
DataLoader worker seeds
model.training flag
dropout state
AMP/autocast
torch.backends.cudnn.benchmark
torch.backends.cudnn.deterministic
torch deterministic algorithms setting
TF32 setting
PyTorch version
CUDA version
cuDNN version
GPU model
```

If nondeterminism is confirmed in A or B, then enable the minimum deterministic controls necessary and repeat the audit.

Any deterministic-setting change must be documented.

---

# 7. Second-Seed Verification

After the root cause is fixed on G60-S3, repeat the final A–D audit on:

```text
G-Load-60 seed1 / 80k
```

No need to repeat the entire forensic process unless seed1 fails.

Purpose:
verify that the fix is pipeline-level, not checkpoint-specific.

---

# 8. Recompute Critical F5.1 Metrics After Fix

Once reproducibility passes, rerun the critical diagnostics for:

```text
G36-S3
G60-S1
G60-S2
G60-S3
G60-S4
NewGL60-L1
NewGL60-L2
NewGL60-L3
```

At minimum recompute:

```text
per-position action centroids
matched-state action distance
between/within position variance
direction cosine / direction correctness metric used previously
position averaging classification
global seed dispersion
local seed dispersion
```

Do not automatically reuse pre-fix values if the evaluation pipeline changed.

---

# 9. R4 Outcome Categories

## `REPRODUCIBILITY_FIXED_PREPROCESS`

Use if divergence originates in input/data/preprocessing and is fixed.

## `REPRODUCIBILITY_FIXED_RUNTIME`

Use if divergence originates in process/CUDA/runtime nondeterminism and is fixed.

## `REPRODUCIBILITY_FIXED_STATE_RESET`

Use if temporal/cache/reset behavior is the cause and is fixed.

## `REPRODUCIBILITY_FIXED_MIXED`

Use if multiple causes contribute.

## `REPRODUCIBILITY_UNRESOLVED`

Use if max difference remains >= 0.001 after the audit.

---

# 10. Gate to Final Dataset Stage

Only if:

```text
R4_REPRODUCIBILITY_PASS = true
```

may the project proceed to the final dataset stage.

The next stage is preplanned as:

```text
FINAL_DATASET_V1
target size ≈ 120–150 episodes
```

---

# 11. Final Dataset V1 Design — Preplanned, Not Yet Executed

Target:

```text
120–150 demonstrations
```

Primary variation:

```text
cup position
small position jitter
small start-state variation
```

Keep stable:

```text
cup type
fill state
camera
task goal
grasp style
lift height
hold duration
place/release pattern
```

Recommended action sequence:

```text
Approach
→ Grasp
→ Lift
→ Hold ~0.8–1.0 s
→ Place
→ Release
```

Hold duration should be deliberately more consistent than in the current dataset.

---

# 12. Position-Coverage Strategy

If using the previous 5 × 3 spatial design:

```text
x = -20, -10, 0, +10, +20 mm
y = -15, 0, +15 mm
```

= 15 position regions.

For 120 episodes:

```text
15 regions × 8 demos ≈ 120
```

Within each region:

```text
~5 standard/small-jitter demos
~2 moderate-jitter demos
~1 harder/edge demo
```

Do not make all repetitions pixel-identical.

Goal:

> learn a continuous position-to-action mapping rather than memorize 15 discrete locations.

---

# 13. Future Dataset-Scaling Study

Once Final Dataset V1 is collected, construct stratified subsets from the same dataset:

```text
N = 36
N = 60
N = 90
N = 120
```

Every subset must still cover all position regions.

Do not use naive first-N sampling.

Use position-stratified sampling.

---

# 14. Future Multi-Seed Scaling Experiment

For each dataset size, initially train:

```text
G-Load
seed1
seed2
seed3
```

Optional seed4 if budget permits.

Measure:

```text
mean performance
std across seeds
worst seed
position sensitivity
position averaging
```

Primary scientific question:

> Does multi-seed variance decrease as demonstration count increases?

If:

```text
seed variance decreases strongly with larger N
```

then data scale may be sufficient to solve most stability issues.

If:

```text
seed variance remains large at N=120–150
```

then proceed to optimization stabilization.

---

# 15. Future Stabilization Branch

Only if large-data seed variance remains large.

Priority:

```text
1. SWA
2. SAM
```

Reason:

### SWA
- low conceptual risk;
- one final model;
- almost no extra inference cost;
- directly compatible with ACT training trajectory.

### SAM
- directly targets sharp minima;
- stronger intervention;
- higher training cost.

Future comparison:

```text
Adam multi-seed
vs
SWA multi-seed
vs
SAM multi-seed
```

Do not introduce these before the large-data baseline is known.

---

# 16. Final Planned Research Story

The intended progression is:

```text
R4:
Fix evaluation reproducibility

↓
Final Dataset V1:
120–150 higher-quality demonstrations

↓
Data scaling:
36 / 60 / 90 / 120

↓
Multi-seed:
seed1 / seed2 / seed3

↓
Question:
Does larger data naturally reduce seed variance?

├─ YES
│  → keep simple baseline
│  → rebuild New-GL
│  → real robot
│
└─ NO
   → SWA / SAM stabilization
   → multi-seed comparison
   → rebuild New-GL
   → real robot
```

---

# 17. R4 Required Outputs

```text
summary.md
validation.md
audit_config.json
frozen_input_manifest.json
same_process_repeat.csv
cross_process_repeat.csv
preprocess_repeat.csv
full_replay_repeat.csv
temporal_state_audit.md
determinism_config.md
seed1_repeat_verification.csv
postfix_f5_1_metrics.csv
builder_review.md
red_team_review.md
proposal.md
```

---

# 18. R4 Completion Criteria

`R4_COMPLETE = true` requires:

1. G60-S3 same-tensor same-process audit complete.
2. Cross-process audit complete.
3. Preprocessing audit complete.
4. Full fresh replay repeat complete.
5. First divergence source identified.
6. Root cause fixed or explicitly unresolved.
7. Final max difference reported.
8. G60-S1 second-seed verification completed.
9. Critical F5.1 metrics recomputed if pipeline changed.
10. Builder Review complete.
11. Red-Team Review complete.

`R4_REPRODUCIBILITY_PASS = true` additionally requires:

```text
max action difference < 0.001
```

for both audited checkpoints under the final pipeline.

---

# 19. Plain-Language Summary

Do not train anything new yet.

First determine exactly why:

```text
same checkpoint
same intended evaluation
```

can differ by:

```text
0.117
```

Audit from the smallest possible unit outward:

```text
same tensor + same process
→ same tensor + different process
→ raw sample + preprocessing
→ complete replay
```

Once this is fixed, stop working on the old 60-episode dataset as the main development dataset.

Then collect a cleaner final dataset of roughly:

```text
120–150 episodes
```

with:
- complete spatial coverage;
- small position jitter;
- consistent grasp/lift/hold/place timing;
- a clear 0.8–1.0 s hold segment.

After that, study:

```text
36 → 60 → 90 → 120 demonstrations
```

with the same 3 seeds.

Only if seed variance remains large on the large dataset should SWA/SAM be introduced.
