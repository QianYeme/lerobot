# T0.1 OC-ACT Spatial Mechanism Diagnostic

## Scope

使用冻结的 `P1_ocfix2_s1/100000`、D1 固定 12-episode Dev split、5748 帧。未重训、未读取 Final Test、未操作真机。D7 使用固定 seed=17 的非等价 local permutation `[1,3,2,4]`；D5/D6/D7 在 raw token 层交换，保持目标槽位 embedding 不变。

## Mechanism evidence

| intervention | h0 arm Δ | all-action Δ | interpretation |
|---|---:|---:|---|
| D1 zero all | 0.03832 | 0.07452 | object branch is used |
| D2 shuffle whole | 0.01141 | 0.02036 | matched object representation matters |
| D3 xy zero | 0.00044 | 0.00078 | direct x/y effect is very weak |
| D4 xy shuffle | 0.00002 | 0.00003 | x/y-to-frame correspondence is nearly unused |
| D5 LR swap | 0.00429 | 0.00727 | horizontal local layout has measurable effect |
| D6 UD swap | 0.01901 | 0.02919 | vertical local layout has stronger effect |
| D7 permutation | 0.01041 | 0.01751 | local slot layout is used |
| D8 global only | 0.02264 | 0.03900 | global token contributes |
| D9 local only | 0.02763 | 0.05250 | local tokens contribute, slightly more than global-only |

The mechanism is therefore not “coordinates are ignored entirely”: P1 uses the object branch and local-token layout. However, the explicit numeric x/y fields themselves have almost no causal effect under D3/D4. The observed T0 Dev L1 improvement should be attributed primarily to object/semantic and local-layout information, not to direct coordinate steering.

## Action-position correlation

The pre-registered all-dimension correlation table is in `action_position_mapping.json`. Expert and predicted h0 actions show strong y correlations on action dimensions 1–3 (up to about 0.63), while x correlations are weaker (about -0.28 to 0.24). Correlation is descriptive only; it does not establish that P1 causally uses the token x/y fields.

## Decision

**PARTIAL.** Q1, Q3 and Q4 are supported; Q2 is not supported as a direct explicit-coordinate mechanism. Token transplant directionality was not run, so PASS criteria are not met. This round stops here as required; a future intervention may target coordinate representation/supervision, but no retraining is authorized by this decision.
