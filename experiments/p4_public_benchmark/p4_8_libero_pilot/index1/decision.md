# decision.md — P4.8-r0 LIBERO Public Benchmark: Controlled ACT vs P1 Pilot

**Status:** FROZEN_FOR_CODEX_EXECUTION (research intent); no execution claimed  
**Date:** 2026-10-30  
**Owner:** Research decision by User + ChatGPT; implementation by **one local Codex** controlling remote Linux server via SSH.  
**Relation:** Independent supplement to ongoing P4.7 Hugging Face release / SO-101 real-robot tests. **Do not pause, restart, overwrite or reinterpret P4.7.**  
**Repository destination (proposal):** `experiments/p4_public_benchmark/p4_8_libero_pilot/index1/decision.md`  
**Allowed deliverables:** `result/` under this experiment, no changes to existing experiment output or checkpoints.

## 0. Purpose, scope, hard constraints

### Scientific question
Can P1's **object-centric visual conditioning** improve policy performance compared with an ACT baseline under identical LIBERO demonstrations and simulator rollouts, without relying on the SO-101 cup setup? Pilot establishes feasibility, not universal superiority.

### Hard boundaries
1. **No real robot** involved in P4.8. Evaluate in simulated closed-loop LIBERO; offline supervised metrics are supplementary.
2. **No direct reuse** of SO-101 P1/ACT weights, eight-dimensional state semantics, six-dimensional joint actions, cup prototype bank, or dataset normalizers. LIBERO requires its own observation/action contract.
3. **Freeze P1 object-conditioning mechanism:** frozen DINOv2-S on main/agentview only, train-only prototypes, 1 global + 4 fixed 2×2 local tokens, adapter, cross-attention with ACT encoder tokens as queries, gated residual and original ACT policy. Any incompatibility must result in `PROPOSAL_REQUIRED`, not silent redesign. Do **not** append load/current artificially.
4. Use **one shared, explicit observation/action adapter specification** for the experiment: LIBERO 8D end-effector state (NOT SO-101 8D), two RGB cameras, 7D end-effector delta + gripper action. ACT baseline and P1 must use identical physical observation channels and control-mode conventions. Only P1 has the frozen DINO object branch.
5. Use a pinned git commit / dataset revision and fixed evaluation initial states/seeds; do not claim numerical comparability to published baselines unless protocol actually matches.
6. **Only one local Codex**, managing the remote server via SSH. No second Codex. Long jobs must survive SSH disconnection using safe, persistent mechanisms, with recorded PID/job IDs, no duplicate jobs.
7. **Do not use Final15 of formal6_merged**. LIBERO is an independent benchmark; keep its evaluation trajectories separate from training.
8. Do not use Hugging Face tokens in logs, scripts, model cards, or repository. Never print secrets. Default `push_to_hub=false`.

## 1. Official starting points — confirm locally, not blindly copy

- LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO
- LeRobot LIBERO integration: https://huggingface.co/docs/lerobot/libero
- Public dataset candidate: `lerobot/libero`, two 256×256 cameras, 8D observation state, 7D actions; metadata and actual available tasks must be inspected. The LeRobot-compatible public data is documented as 1,693 episodes across 40 tasks (not the 130-task environment total).
- Linux and headless MuJoCo `MUJOCO_GL=egl` documented. GPU/GL library support is **not assumed**; test it.
- LIBERO evaluation controls include `--env.type=libero`, `--env.task=libero_object`, `--env.task_ids`, `--eval.n_episodes`, `--env.control_mode`. Consult **the installed repo version's `--help`** before any commands; LeRobot APIs may differ by version.

## 2. Execution phases, each with a blocking gate

### Phase 0 — Inventory & isolation (NO training)
- Inspect current SSH server tasks, P4.7 status, CPU/GPU utilization, available hours, free disk and conda/venv, LeRobot git SHA, CUDA/MuJoCo versions.
- Create **separate virtual environment/working tree** or otherwise demonstrate dependency isolation from P4.7. Do not upgrade global LeRobot or replace training scripts in place.
- Confirm benchmark datasets / pretrained DINO cache / disk footprint and license. Record `environment_lock.md`, `server_resource_snapshot.json`, `source_provenance.json`.
- If current P4.7 tasks lack spare resources, execute only low-resource static checks; queue later pilot safely. Do not preempt.

**Gate G0:** Independent environment, source commits, data revisions and resource budget reproducible. Otherwise STOP + PROPOSAL.

### Phase 1 — Sim/data feasibility (NO model training)
- Install LIBERO integration in isolated Linux env as documented; use headless EGL if supported.
- Test environment reset, two cameras, state/action schema, a **trivial safe action**, simulator step, render, episode reset, reward/success reporting, deterministic seeded init-state pairing. Do not assert it works until executed.
- Inspect train examples and metadata for a candidate `libero_object` task whose demonstrations are actually available with matching task IDs / camera frames. Verify explicit mapping from dataset task annotation to simulator task and instruction; do not assume suite names imply available training data.
- Confirm action scale, control mode, timebase, camera channel names, image normalization, frame rates, action semantics, padding behavior, end-effector vs joint space.
- Compare LIBERO baseline action/state normalization to required policy normalization. Produce **sample-by-sample sanity report**: frames, dimension, time alignment, number episodes/frames for selected task, no cross-episode mixing.
- Collect 3 very short simulator smoke rollouts (untrained/random/zero-action) to verify complete evaluation loop; success rate not an experimental result.

**Gate G1:** LIBERO environment and dataset task matches; rollout metrics exist; policy input/output contract frozen. Otherwise STOP + PROPOSAL.

### Phase 2 — Freeze two model identities and prototype semantics

**ACT-LIBERO (primary baseline):** Native LeRobot ACT backbone/policy with only legitimate LIBERO dimensional/camera/task adaptation; no DINO/object tokens/force/load. Freeze training hyperparameters, action horizon, observation history, image backbone and runtime settings. Distinguish (a) library defaults and (b) controlled overrides. If default native ACT and matched-compute ACT differ, report both separately; do not label adapted policy "fully default" without proof.

**P1-LIBERO:** ACT-LIBERO's **exact same ACT subnetwork and train configuration** plus P1 frozen DINO object-token cross-attention/gated residual. Gate alpha initial zero. Preserve all pre-existing P1 object-path semantics except the **domain-specific prototype bank**. The SO-101 cup prototypes are forbidden here. Prototype construction needs a frozen, pretraining-independent rule using **only selected LIBERO task training images**. Specify in `prototype_spec.md` exactly:
- how relevant object regions/patches are identified without using validation frames, simulator privileged segmentation at eval time, manually inspecting test scenes or test leakage;
- 4 prototype vector selection method, seed, source frame/patch IDs, feature dimensionality, DINO preprocessing and hashes;
- whether training-only object metadata/segmentation is used; if unavailable, **stop and propose** a deterministic unsupervised object-prototype strategy, with separately reviewed justification. Do not silently substitute arbitrary patch clustering or a generic pretrained object detector as if it were original P1.
- invariant 1 global + 4 local fixed 2×2 tokens, cosine prototype matching, tau=0.20, local coords/scores, adapter, encoder-token Q and object-token K/V, learned residual gate; verify DINO sees **once** normalized raw camera RGB, while ACT image preprocessing remains unchanged.

Use separate namespace and code-level architecture identity tests. Capture structured `architecture_diff.md` with code references and parameter counts. Do not change normal robot P1 architecture while adapting to LIBERO.

**Gate G2:** Architecture review and prototype provenance PASS, both ACT and P1 pass batch forward/backward/normalization smoke, no information leakage. Otherwise STOP + PROPOSAL.

### Phase 3 — Pairwise single-task feasibility experiment

- Select and freeze **one** LIBERO-Object task with available demonstrations before seeing either model's performance. Record exact task ID, text, source files, number episodes, selection reason.
- Train ACT-LIBERO seed1 and P1-LIBERO seed1 on **identical demonstrations**, same sequence/padding, same input modalities, same action labels and output scale, same batch size, optimizer/schedule/steps, same augmentation and nominal runtime budget. The frozen DINO adds compute/parameters; report cost and parameter count separately.
- Pilot step budget: **10k updates/model** as a feasibility target, not a performance claim. First run 100-step smoke per model, compare serial vs two concurrent process throughput; choose concurrency only if **aggregate useful throughput** improves, without risking P4.7. Monitor OOM, latency, disk, overfit and data-loader bottlenecks.
- Save checkpoints at 2k/5k/10k and optimizer/RNG states. Do not declare 10k sufficient to compare final quality. Stop or adapt budget only with written proposal if clear nonconvergence.
- Run matched-seed **closed-loop simulations**, primary pilot **10 episodes per task per model** (same eval init states), with hard reset for official comparability. Record success/fail counts, exact init IDs, max steps, rollout videos and failure stages. Report offline action errors separately and caution that closed-loop distribution differs from teacher-forced samples.
- Run `P1 object on/off` diagnostic **on one frozen P1 checkpoint** without updating weights, as evidence of model sensitivity only—not a causal training ablation. A true module ablation later requires a controlled retraining design.
- If policy errors cause dangerous numerical simulator behavior, stop simulation and log; there is no physical robot.

**Gate G3 / pilot acceptance:** Both policies trained, reloadable; at least 10 paired simulator rollouts each; traceable success metrics; matching observation/action contract; prototype provenance; no data leakage. Negative results count as completed scientific results. If a model cannot train, mark pilot incomplete.

### Phase 4 — conditional full benchmark proposal (DO NOT auto-run)
If G3 passes and empirical throughput/resource cost is acceptable, return a `proposal_P4_8_1_libero_object_full.md` for user approval, NOT autonomous launch:
- 10 LIBERO-Object tasks (verify data and task coverage), train/eval ACT vs P1 with matched recipes and >=3 seeds when feasible; task success aggregated per task, macro-average, seed variance / confidence intervals.
- Use **>=10 matched rollouts/task/seed** for screening; for publication-quality claims plan higher episode counts, fixed init-state protocol, confidence intervals, and report true hardware costs. Cross-check official LIBERO published protocol.
- Optional LIBERO-Spatial only after Object established. Diffusion optional at same public dataset with appropriately disclosed architecture/training cost.
- Evaluate object path ON/OFF with a properly controlled training ablation if a paper claim about the module's contribution is intended.
- **New-GL force module is out of scope**: LIBERO lacks equivalent load/current, so this benchmark tests P1 visual architecture portability only.

## 3. Metrics and evaluation integrity

Primary: LIBERO closed-loop task success count/rate per task / paired seed / model. Secondary: grasp-like phase if measurable without privileged future info; rollout length; collision or reset failure; ACT vs P1 training wall-clock, samples/sec, GPU VRAM, effective FLOPs if measured; inference p50/p95 latency; #params; offline action MAE with exact normalization units; DINO and prototype matching diagnostics. Do not infer improvement from ON/OFF action magnitude alone.

No cherry-picking checkpoint on final eval rollouts. Select checkpoints using training/held-out validation demonstrations or an explicitly designated *development* rollout set, then freeze before final eval. If only 10 rollouts are available, label them pilot/exploratory rather than unbiased final benchmark. Report all negative outcomes.

## 4. Single-Codex SSH execution workflow and result contract

**One local Codex:**
1. Read `decision.md`; audit P4.7 tasks. Implement isolated P4.8 scripts in separate worktree; do only G0–G1 initially.
2. Once G0–G1 pass, freeze G2 artifact and verify matching ACT/P1 model definitions; stop if prototype specification requires researcher decision.
3. If G2 passes, launch independent persistent remote jobs; store PIDs, effective configs, queue state, timestamps and update step in `result/server_jobs.json` and logs.
4. SSH disconnect must not kill jobs; a second invocation must detect existing jobs/checkpoints and avoid duplication. The local Codex remains free to work on P4.7 real robot tasks when server jobs run remotely.
5. `result/` must include `summary.md`, `environment_lock.md`, `source_provenance.json`, `dataset_task_mapping.md`, `data_contract.md`, `prototype_spec.md`, `architecture_diff.md`, `training_manifest.csv`, `eval_rollouts.csv`, `eval_summary.csv`, `cost_profile.csv`, `checkpoint_manifest.csv`, `object_path_diagnostics.csv`, `builder_review.md`, `redteam_review.md`, `blocking_proposal.md` (if any), and hashes/locations. Missing evidence must be marked explicitly, not fabricated.
6. The user and ChatGPT own research decisions; Codex MUST NOT rewrite frozen scientific intent or silently change the budget/algorithm/task. Report conflicts as a proposal.

## 5. Explicit success/failure disposition
- `ENV_BLOCKED`: isolated simulator/headless install fails or clashes with P4.7.
- `DATA_CONTRACT_BLOCKED`: public dataset and selected env task cannot be aligned.
- `PROTOTYPE_SEMANTICS_BLOCKED`: cannot create train-only object prototypes without unverifiable leakage or altering core P1 meaning.
- `PILOT_READY`: ACT and P1 trained and evaluated with reproducible paired simulator results.
- `PILOT_INCONCLUSIVE`: experiment ran but comparisons invalid (mismatched recipe, too few paired rollouts, nonconvergence, missing checkpoint evidence).
- `PILOT_NEGATIVE`: valid comparison finds no P1 gain; do not suppress.

**Immediate next task:** inventory and simulator smoke in isolation, then freeze the selected task and prototype construction protocol. No full-suite training until the next approved decision.
