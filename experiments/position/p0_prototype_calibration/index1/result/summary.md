# P0 Cup Prototype Calibration — result

Date: 2026-10-11  
Execution: remote RTX 5090, CUDA, frozen `dinov2_vits14`  
Decision: `decision_P0_CUP_PROTOTYPE_CALIBRATION.md` (read-only)

## Verdict

**P0 = FAIL for formal training.** The prototype bank is retained as a diagnostic candidate only and must not be used by formal OC-ACT training or deployment.

## What was evaluated

- Frozen train-only references: 8 episodes (`0, 6, 10, 17, 22, 35, 40, 55`), frame 15.
- Independent train split check frames: 12 episodes (`2, 7, 13, 18, 23, 25, 30, 36, 42, 45, 50, 56`), frame 15.
- Coarse boxes were visually reviewed; no dev or final-test data was read.
- Candidate bank sizes 1, 4, and 8 were compared using the pre-frozen selection rule. The default 4 prototypes remained selected.

## Measured evidence

| Gate signal | Result |
|---|---:|
| peak inside coarse cup box | 12/12 (1.000) |
| peak inside box or one neighbouring patch | 12/12 (1.000) |
| cup-vs-corner proxy | 12/12 (1.000) |
| inside/outside similarity ratio (mean) | 2.786 |
| soft-center error, median / P90 | 78.28 / 97.77 px |
| inside softmax mass (mean) | 0.0508 |
| object-token shape | `[12, 5, 387]`, finite |

The important contradiction is that the similarity **peak** is in the cup region, while the untempered M0 softmax pooling produces a diffuse soft center near a nearly fixed image-center location. The soft center therefore does not follow cup motion across the check frames. This violates the P0 requirement that the object signal be spatially useful, even though the coarse peak and corner shortcut checks pass.

## Artifact/provenance

`cup_prototype_bank_v1.pt`, `metrics.json`, `prototype_provenance.json`, overlays, reference manifest, and coarse-box review sheet are complete. The bank is marked `FORMAL_P0_CANDIDATE` for traceability, but this result supersedes that label for eligibility: `formal_training_eligible=false`.

## Required stop

Do not start formal ACT/OC-ACT training, do not read dev/final-test to tune this bank, and do not deploy it. A new decision is required for a spatial-pooling/object-token fix (for example temperature or localized pooling) followed by a fresh train-only P0 calibration.
