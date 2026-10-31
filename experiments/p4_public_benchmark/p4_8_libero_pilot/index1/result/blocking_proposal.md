# Reviewable G2 proposal — no training launched

## Blocking evidence and required decision
P4.8 Phase2 says that if training-only object metadata/segmentation is unavailable, stop and propose a deterministic unsupervised object-prototype strategy, separately reviewed. annotation_and_boundary_audit.json confirms no region supervision; dataset sample previews show multiple distractor objects. No actual P1-LIBERO bank can yet be traced to the target. G0 and G1 technical checks passed; model campaign is not authorized past this failed gate.

## A. Deterministic training-interaction prototypes (requested unsupervised proposal; NOT implemented)
1. First freeze a task-specific Train/Dev split, suggested35/9 of44 episodes by seed1; keep eval-init states and Dev images out of prototype construction. Freeze all IDs/hashes before any model performance.
2. Choose four Train episodes at fixed quantile indices0,11,23,34 of sorted Train IDs. Locate strongest closure event from training gripper-qpos derivative with a fixed5-frame median and deterministic earliest tie-break. Use only fixed short before/after closure windows in these training trajectories.
3. In corrected raw main RGB, compute deterministic Farneback flow (pyr_scale.5,levels3,winsize15,iterations3,poly_n5,poly_sigma1.2,flags0). Rank DINO patches by post-closure minus pre-closure residual motion, combined with flow-tracked DINO feature consistency. Static distractors should score lower; ties use episode/frame/patch index. No object detector,external masks,Dev/test scenes or simulator eval segmentation.
4. Select one positive-ranked track from each fixed episode; pool its384D DINO features using the original prototype-bank aggregation/normalization semantics after code review. Record exact episode/frame/patch track IDs,score maps,DINO preprocessing/weights and final SHA. Keep the frozen token mechanism unchanged.
5. Major falsification risk: moving robot/gripper may outrank target. This is an interaction-prototype heuristic, NOT target GT. Report candidate Train-only visual evidence for review; if it is predominantly robot or absent, stop. Do not train first and redefine the selection rule after seeing performance. Do not present it as numerically identical supervision to the SO101 cup bank.

## B. Small explicit Train-only manual object ROI (alternative requiring approval)
After the same split freeze, human-label four fixed training frames with target-positive polygons/patches. Unknown/occluded pixels stay unknown, never negative/background. Use original pooling rules; record annotation provenance and manual effort. This gives a clearer target bank but adds human supervision to P1; report that extra prior and avoid calling it a purely architecture-only comparison. No annotation has been created by this task.

## Common recipe review item
Freeze native20Hz relative controller, one model action per simulator step for both methods,hard reset and shared init IDs; retain dataset10fps metadata and disclose timing provenance limitation. Alternatively investigate original conversion physical-time provenance first. No silent controller-rate/action-scale or timestamp changes.

## Parallel plan after approval
Run only the authorized100-step serial/two-concurrent throughput probes; choose concurrent training only if aggregate useful throughput improves and memory is safe. ACT and P1 each10k seed1,2k/5k/10k full checkpoints. Then distribute the fixed10 paired-init rollouts/model over2–4 workers based on measured simulation throughput/VRAM; one unique(model,init_id) record and hard reset, no new seeds or extra final episodes. P1 object ON/OFF sensitivity uses one frozen checkpoint and is not a retraining ablation. No full-suite auto-run.
