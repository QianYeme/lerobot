# T0.2 Token Transplant Directionality

## Scope and validity

This is a Dev-only diagnostic on frozen `P1_ocfix2_s1/100000`; it does not retrain, read Final Test, or operate the robot. The same base Top/Wrist RGB and 8D state are used for each pair; only object tokens are transplanted. Pair selection was frozen before counterfactual action inference: confidence >= 0.20, pre-release timing fraction <= 0.75, state-standardised nearest matching with time difference <= 0.20, and 20 deterministic pairs per group (seed 17). No phase labels exist in Formal4, so the timing rule is explicitly recorded rather than presented as human phase ground truth.

## Result

The outcome is **PARTIAL (one-axis mechanism evidence)**. Longitudinal transplant is directionally consistent for whole/local/global tokens (0.90/0.85/0.90), exceeds the 0.70 reference, and the large-position response is larger than same-position control (whole arm delta 0.0691 vs 0.0183). Random-unmatched longitudinal control is only 0.15. Horizontal directionality does not pass (whole/local/global: 0.20/0.00/0.20); therefore the model cannot be claimed to have reliable two-axis steering.

Whole-token and global-only responses are larger than local-only for large longitudinal pairs, while local-only still preserves the longitudinal sign. Gripper response is reported separately; the dominant longitudinal effect is not a gripper-only change.

## Fixed interpretation

The frozen P1 OC-ACT representation contains evidence of causal longitudinal object-conditioned steering, but the horizontal channel remains unreliable and is confounded by weak spatial sensitivity in T0.1. This is not a PASS for a complete spatial mechanism. Do not enter Confirmation or Final Test from this result; the next research decision should target a single horizontal-steering intervention or an improved position representation.

See `metrics.json`, `controls.json`, `pair_manifest.json`, `transplant_results.json`, and the PNG figures in `figures/` for raw evidence.
