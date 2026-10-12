# P0.1 Cup Spatial Pooling Fix — result

Date: 2026-10-11  
Execution: remote RTX 5090/CUDA, frozen `dinov2_vits14`  
Decision: `decision_P0_1_CUP_SPATIAL_POOLING_FIX.md` (read-only)

## Verdict

**P0.1 = PASS.** The only tested intervention was temperature-scaled soft pooling. The selected configuration is `tau=0.20`, chosen as the largest frozen candidate that satisfies all gates. It is a calibration artifact, not a formal policy checkpoint.

## Frozen scope

- Same P0 four-prototype bank, eight reference frames, twelve calibration-check frames, coarse boxes, DINO model, patch coordinates, and five-token layout.
- Candidate temperatures were exactly `1.00, 0.20, 0.10, 0.05, 0.02`.
- No dev or final-test data, policy training, detector/SAM, prototype rebuilding, or robot execution.

## Temperature comparison

| tau | median / P90 error (px) | Pearson x/y | Spearman x/y | effective patches | top-1 weight | peak / corner gates |
|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 78.28 / 97.77 | .719 / .799 | .476 / .727 | 251.3 | .0069 | PASS / PASS |
| **0.20** | **24.75 / 32.73** | **.964 / .931** | **.888 / .895** | **120.1** | **.0422** | **PASS / PASS** |
| 0.10 | 5.50 / 8.09 | .990 / .955 | .972 / .937 | 26.7 | .1128 | PASS / PASS |
| 0.05 | 5.93 / 7.65 | .985 / .944 | .972 / .916 | 11.0 | .2125 | PASS / PASS |
| 0.02 | 9.52 / 16.12 | .947 / .894 | .895 / .916 | 4.8 | .4122 | PASS / PASS |

The lower temperatures improve pixel error further, but the frozen selection rule prefers the **largest passing temperature** to retain a diffuse soft representation and avoid unnecessary one-patch collapse. `tau=0.20` is therefore selected; `tau=0.10` remains a diagnostic sensitivity result, not the selected configuration.

For `tau=0.20`, the median coarse-box diagonal is 131.52 px, so the gates are 39.46 px (median) and 59.18 px (P90). The measured 24.75/32.73 px passes both. Predicted center spans are 26.82×17.40 px versus target spans 50.0×33.5 px; correlations are strongly positive and the center is no longer fixed at the image center.

## Artifacts

- `object_pooling_config_v1.json` — selected pooling configuration.
- `metrics.json` — all frozen tau diagnostics.
- `tracking.json` and `figures/soft_center_tracking.png` — target/predicted center tracking.
- `figures/temperature_overlays/` — representative overlays for every tau.

P0.1 makes the object representation spatially usable under the frozen calibration contract. Formal ACT/OC-ACT training is still a separate decision and has not been started.
