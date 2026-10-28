# decision.md — P4.0 formal6_merged 数据验收与划分冻结

## 0. 决策头（研究语义冻结）

- Experiment ID: `P4_0_FORMAL6_DATASET_AUDIT_AND_SPLIT_FREEZE`
- Revision: `r0`
- State: `FROZEN / NOT_EXECUTED`
- Dataset: `QYyyyyyyy/formal6_merged`
- Dataset URL: https://huggingface.co/datasets/QYyyyyyyy/formal6_merged
- User-reported collection: **120 episodes**（待程序核验）
- Model to serve in next stage: **P1_OCACT**, historical best architecture/checkpoint kept unchanged.
- This round: **OFFLINE DATA AUDIT ONLY**. `TRAINING=NO`, `REAL_ROBOT=NO`, `NEW_COLLECTION=NO`.
- Main problem: verify that the new left-source/right-target demonstration data are complete, usable and appropriately partitioned before training.
- Responsible roles: ChatGPT/user freeze research design; Codex executes and reports; Codex cannot revise this decision and must use `PROPOSAL.md` to request changes.
- Working location: create an independent `experiments/p4_final_training/index0/` or equivalent experiment directory, never mutate prior datasets/results.

> **Grounding warning:** I have not independently verified this dataset's actual Hugging Face schema, file count, per-episode lengths or quality. The known 120 episodes and task geometry come from user reports and prior P3 decision. Execution must confirm all rather than assume them.

## 1. Inputs and frozen scientific assumptions

1. New task: transparent half-filled plastic cup, **left Source Area → right Target Zone**. Grasp in a broad left area, lift, airborne hold, transport right, place and release in a smaller fixed right zone.
2. Planned geometry (candidate, do not silently claim measured): source about **120 × 80 mm**, target about **60 × 60 mm**, gap about **50–100 mm**; actual dimensions and orientation require collection setup/metadata confirmation.
3. Historical collection plan was **15 logical source regions × 8 valid episodes = 120**, with desired spatially stratified **Train90 / Dev15 / Final15** (6/1/1 per region). These are *targets*, not assumed facts.
4. Existing P1 state is **8D** = 5 arm joint positions + `gripper.pos` + `gripper.load` + `gripper.current`; action **6D** = 5 arm + gripper command. Confirm exact *stored column mapping*, names, units and normalization. Do not force schema adaptation without a proposal.
5. P1 includes original ACT top and wrist branches, frozen DINOv2-S on top image only, four Train-only cup prototypes, object tokens, cross-attention and gated residual. This stage does not build prototypes or train P1.
6. Force auxiliary work is closed (`FORCE_AUX_STOP`); do **not** remove existing load/current from P1.
7. Historical video/CSV *exports* may be unsynchronized; do not force alignment of unrelated recordings. LeRobot-native frames/videos may have a shared episode/frame index; verify actual timing before exploiting it.

## 2. Exact research questions

**Q1:** Does `formal6_merged` contain 120 distinct, decodable, physically meaningful full-task demonstrations?

**Q2:** Are image/state/action streams compatible with the frozen P1 interface without semantically changing it?

**Q3:** Does the *actual* initial cup distribution cover the intended left-hand two-dimensional source space, rather than merely having tape marks?

**Q4:** Are grasp, physical lift, airborne hold, rightward transport, target placement and release represented clearly and consistently?

**Q5:** Can a non-leaking stratified Train/Dev/Final assignment be frozen with all 15 source regions represented?

No model comparison is performed this round.

## 3. Execution rules and dataset snapshot

1. Locate the dataset via the supplied repo ID. Use a **pinned immutable dataset revision/commit SHA** and record it in `dataset_identity.md` and every downstream config. If remote access is not possible, inspect an existing complete local snapshot and record its provenance; never substitute an older dataset.
2. Read-only inspect original raw dataset; **do not rewrite, reorder, filter or upload** the original HF dataset.
3. Verify metadata (`info.json`, `meta/episodes*`, feature declarations, video/file structure, index semantics), if present, and enumerate actual episodes from data independently of metadata.
4. Record library/runtime versions and content hashes where practical; distinguish metadata claims from checks actually executed.
5. If remote access/auth is blocked, report `BLOCKED_DATA_ACCESS`, missing information and specific minimal remedy. Do not manufacture summary statistics or a split.

## 4. Audit A — integrity and field contract

For every actual episode, inspect and report:

- Episode identity and frame count; zero/missing/duplicated/non-monotonic episode/frame indices; timestamp monotonicity and time-step distribution; inferred FPS vs declared FPS.
- Top and wrist/`gripper` camera keys as actually present; frame decoding on representative early, middle and late frames; resolution/codec/missing frame anomalies.
- `observation.state` dimensions and semantic order; `action` dimensions and order; load/current column availability and sign/value ranges; dtypes, NaN, Inf and outlier/discontinuity checks.
- Action–state conventions, actuation units and whether actions are absolute positions; verify against historical P1 without guessing.
- Possible repeated/corrupted episodes and significant reset inconsistencies; detect exact duplicates with hashes or feature checks and flag near duplicates for review.
- Camera and hardware/teleop session changes identifiable from metadata or videos; do not claim stability that cannot be observed.

Output `schema_and_interface_audit.md` with explicit `P1_8D_6D_COMPATIBLE = PASS/FAIL/UNKNOWN` plus reasons. A schema mismatch is **not** permission for Codex to change P1.

## 5. Audit B — episode task quality

Verify the task using **actual video evidence**, not only action timing. Assess each episode for:

`START_IN_SOURCE`, `APPROACH`, `PHYSICAL_GRASP`, `PHYSICAL_LIFT`, `AIR_HOLD`, `RIGHTWARD_TRANSPORT`, `TARGET_PLACEMENT`, `RELEASE`, `COMPLETE_TASK`.

- Prefer efficient automated pre-screening and representative contact sheets, but **ambiguous physical events must be reviewed**, not inferred from commanded gripper position.
- Distinguish `PASS`, `FAIL`, `UNCERTAIN` for every episode. Uncertain is neither pass nor fail.
- Track approximate hold duration where defensible; target training guideline **0.8–1.0 s**, but natural variation does not automatically make an episode invalid. Report distribution, not arbitrary rejection.
- Record transport completion, obvious spills, slips, collisions, operator recoveries, premature retreat, missing end-of-episode phase.
- Do **not** assume the Target 60×60 mm border means cup *center* in the border implies the entire 50 mm base is contained. Specify evaluation geometry separately and defer strict physical target-success thresholds if the setup cannot establish them.
- No requirement to create full frame-level temporal labels for all 120 episodes. Use coarse episode-level quality plus targeted detailed inspection where needed.

Provide `episode_qc.csv` with columns `episode_id, readable, start_in_source, grasp, lift, hold, transport_right, target_place, release, complete_task, qc_status, uncertainty, evidence_note` (allow additional fields). Never silently discard recorded trials.

## 6. Audit C — source position coverage and session confounds

1. Search for collection manifest / region labels / measured cup XY coordinates. Preserve genuine annotations and coordinate reference.
2. If region labels exist, report count and actual valid counts for **R01–R15**; verify purported 8 valid/region.
3. If no region labels exist, attempt image-based *approximate* starting cup localization only if robust and visually verifiable; **do not automatically turn noisy inferred coordinates into ground-truth labels**. Emit `REGION_LABELS_UNVERIFIED` if not reliable.
4. Produce source-position scatter plot/heatmap in image coordinates; annotate that pixels are not millimeters unless calibrated. Compare within-region variations and edge/center coverage.
5. Check for near-identical start positions and their potential leakage across splits, especially if repeats occurred across sessions.
6. Check whether collection order/session (camera, lighting, tape, robot start/home, fill level) correlates with region. Document discovered drift.
7. Separately inspect whether final placement target remains physically fixed and visually identifiable; avoid imposing an unsupported millimeter accuracy requirement.

No need to track exact XY in millimeters if unsupported; **uncertainty must remain explicit**.

## 7. Audit D — movement and hold distributions

Provide per-episode and aggregate summaries for total episode length and defensible task phases. Use native frame/time mapping only where verified. Retain histogram/summary for:

- time before first grasp; grasp-to-physical-lift latency;
- Lift duration / height *qualitative* category (do not invent heights);
- airborne hold duration, and whether hold is distinguishable from transport;
- transport phase duration and movement direction;
- Target place/release time, trailing idle fraction, unusual pauses;
- gripper.pos, load/current trajectory statistics (diagnostic, **not** renewed reactive-controller study).

Report completion and uncertainty rate per region if possible. A low-confidence phase is labeled unknown, not scored as a confirmed event.

## 8. Split freeze (conditional, not automatic)

Target is **90 Train / 15 Dev / 15 Final**:

- If all 15 genuine regions have >=8 *valid* episodes, construct 6/1/1 **per region**.
- Prefer grouped spatial separation when possible: near-duplicate placements/near-identical trajectories belong in same split or are excluded from held-out exact-position interpretation.
- Balance sessions and collection order to minimize confounding. Fix a deterministic, documented selection algorithm and seed before selection. No cherry-picking by eventual model error.
- If 120 recordings include invalid episodes, **do not call the remaining set Train90/Dev15/Final15 by fiat**. Report achievable counts and propose remediation or a revised split for scientific approval.
- If region labels are absent/unverifiable, generate a **provisional** split proposal, `SPLIT_FREEZE_BLOCKED`, not a fictional stratified split.
- `Final15` is strictly held out: no checkpoint tuning, prototype construction, data augmentation calibration or hyperparameter selection using Final content. Any limited inspection for QC/split assignment must be documented; Final labels/quality audit is not model evaluation.
- Store explicit immutable episode ID lists and dataset commit ID; report SHA-256 for manifest content and overlap checks; assert zero episode overlap, and flag spatial-neighbor/duplicate risks.
- Make separately versioned `train90_ids.txt`, `dev15_ids.txt`, `final15_ids.txt` only when freeze is valid. Never change them opportunistically after seeing training outcomes.

## 9. Train-only prototypes and P1 contract preflight

**Do not actually build prototypes yet**. Write a plan specifying:

- Train-only source episode/frame policy for the **four cup prototypes**.
- Strict prohibition against using Dev/Final images (including indirect use from generated contact sheets selected based on later model errors).
- Exact P1 expected input names/shapes and dataset field mapping; any mismatch is a blocking item/PROPOSAL.
- Note changes needed to train/evaluate on new dataset, without implementing architectural changes.

## 10. Outcome gates (priority ordered)

1. `DATASET_BLOCKED`: inaccessible, corrupt, incompatible state/action or insufficient evidence. Stop. Output precise blocker report.
2. `DATASET_AUDITED_SPLIT_BLOCKED`: raw data usable, but spatial labels, QC or counts do not support frozen 90/15/15. Provide explicit alternative **proposal only**.
3. `DATASET_READY_FOR_P4_1`: 120 unique readable/usable task episodes (or user-approved documented deviations), valid compatible interface, defensible spatial coverage and frozen non-leaking split.

**Do not conflate “120 recordings” with “120 QC-pass valid demonstrations.”**

## 11. Limits / no-go

Codex shall NOT:

- Start training or offline model inference sweeps; no P1 checkpoint evaluation in this round.
- Operate the physical robot, collect more data, adjust cup or move tape.
- Change P1 architecture, state to 6D, visual branches, chunking, prototypes, or Force auxiliary.
- Create or modify the HF dataset without explicit separate approval.
- Automatically discard uncertain episodes, fabricate labels, infer millimeter XY from uncalibrated video, force alignment on unsynced exports.
- Claim quantitative real-robot success or learned spatial generalization from dataset QC alone.

## 12. Result package — handoff to ChatGPT

Save under `experiments/p4_final_training/index0/result/`:

```text
summary.md
validation.md
dataset_identity.md
schema_and_interface_audit.md
episode_manifest.csv
episode_qc.csv
source_region_coverage.csv
source_position_analysis.md
source_position_scatter.png
phase_duration_statistics.csv
phase_duration_analysis.md
collection_session_confounds.md
split_method.md
split_manifest.csv
train90_ids.txt                 # only if freeze gate passes
dev15_ids.txt                  # only if freeze gate passes
final15_ids.txt                # only if freeze gate passes
split_hashes.txt               # only if freeze gate passes
prototype_source_plan.md
training_preflight.md
builder_review.md
red_team_review.md
PROPOSAL.md
```

Use `N/A - reason` for genuinely unavailable files instead of fictional metrics. `summary.md` must state `TRAINING_RUN=false`, `ROBOT_RUN=false`, `DATASET_MODIFIED=false`, exact dataset SHA (if available), quality pass/uncertain/fail counts, spatial evidence, and outcome gate.

## 13. Builder Review

Check that actual dataset version and episode count were verified, all 120 episodes inventoried, feature dimensions measured, videos spot-checked, state/action contract compared against P1, QC counts reported with uncertainty, source coverage assessed, freeze criteria enforced, Train-only prototype restrictions recorded, and zero training/robot operations performed.

## 14. Red-Team Review

Ask: Could label extraction just reflect a drawn grid rather than cup coordinates? Are repeated placements shared across splits? Does video actually verify Lift/Hold? Are task success claims justified by visibility? Is the collection session confounded with region? Does Target boundary interpretation match the 50mm cup base? Are P1 state/action units actually consistent? Are Final15 images indirectly used to choose prototypes? Were any unverified facts quietly promoted into measurements?

## 15. Completion and next decision

`P4_0_COMPLETE=true` means the audit and reviews were executed and the true outcome gate was assigned; it does **not** necessarily mean dataset approved.

If `DATASET_READY_FOR_P4_1`, user + ChatGPT review and **then** generate `P4.1 — fixed P1 training recipe / pilot budget` as a separate decision. If blocked, present minimal remedy proposal before any training. No autonomous modification to `decision.md` by Codex.

## 16. What this decision does (for user)

This decision **does not train a model**. It checks whether `formal6_merged` is really a clean, appropriately varied 120-episode dataset for the frozen P1, and either freezes a defensible Train90 / Dev15 / Final15 or clearly states why it cannot. Its only purpose is to prevent a rushed, scientifically uninterpretable final training run.
