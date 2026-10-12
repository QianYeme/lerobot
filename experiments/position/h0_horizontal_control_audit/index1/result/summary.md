# H0 Horizontal Control Signal Audit

## Scope
Read-only diagnosis on fixed Formal4 Train/Dev episodes. No training, architecture change, Final Test, or real-robot operation was performed.

## Frozen method
P0.1 x_global was used as the target. Because no reliable SO101 URDF/Jacobian is present, Method B fits a Train-only ridge from five arm joint target deltas to x_global. The mapping is frozen before model re-evaluation.

## Evidence
- Selected reach/fine-alignment frames: Train 1501, Dev 556
- State-matched pairs: Train 33, Dev 2; Dev pair count is too small for an independent gate.
- H=1 Dev: sign accuracy 0.401, Pearson -0.171, Spearman -0.285.
- H=3 Dev: sign accuracy 0.448, Pearson -0.142, Spearman -0.236.
- H=5 Dev: sign accuracy 0.475, Pearson -0.102, Spearman -0.164.
- H=10 Dev: sign accuracy 0.514, Pearson 0.041, Spearman -0.006; this is near chance, not a stable steering signal.

## Model re-evaluation
Existing token-transplant outputs were projected through the frozen mapping. These are mechanism diagnostics, not new model selection metrics; see `model_re_evaluation.json` for all variants.

## H0 decision
**H0-C / unresolved: do not attribute the observed initial centering to a single action dimension.** The multi-joint empirical mapping is not independently validated on Dev (only 2 matched pairs) and the frame-level temporal signal is weak or sign-inconsistent. Stop architecture expansion based on this hypothesis until the dataset/task signal is repaired or a reliable kinematic calibration is available.

## Next safe step
Improve the diagnostic boundary: obtain a trustworthy SO101 kinematic calibration or collect a deliberately varied horizontal trajectory set with explicit reach-phase labels, then rerun H0. Do not start another architecture/training campaign from this result alone.
