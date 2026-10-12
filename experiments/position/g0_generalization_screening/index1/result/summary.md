# G0 Position Generalization Screening

`POSITION_GENERALIZATION_SHORTLIST_READY = TRUE`

## Scope
Fixed Formal4 Dev 12 episodes only. No retraining, no architecture change, no Final Test, no real robot.

## Candidate summary (h0 teacher-forced error)

| Rank | Model | Overall L1 | Arm L1 | Gripper L1 | Left | Center | Right | Near | Far | Edge | Center-bias | Object/spatial usage | Params | Latency |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | B3_ACT | 0.08547 | 0.09053 | 0.06015 | 0.08441 | 0.08937 | 0.09782 | 0.07192 | 0.11402 | 0.08770 | higher separation | prior mechanism evidence; not rerun | not recorded | not recorded |
| 2 | B2_ACT | 0.08828 | 0.09317 | 0.06381 | 0.08501 | 0.09293 | 0.10158 | 0.07586 | 0.11514 | 0.09107 | higher separation | prior mechanism evidence; not rerun | not recorded | not recorded |
| 3 | P1_OCACT | 0.08364 | 0.08851 | 0.05931 | 0.07772 | 0.08688 | 0.10094 | 0.07181 | 0.11186 | 0.08683 | higher separation | prior mechanism evidence; not rerun | not recorded | not recorded |
| 4 | P3_FIXED_SPATIAL | 0.09713 | 0.10109 | 0.07729 | 0.09298 | 0.10063 | 0.10967 | 0.07629 | 0.12631 | 0.09833 | higher separation | prior mechanism evidence; not rerun | not recorded | not recorded |
| 5 | P4_JOINT_SPATIAL | 0.12671 | 0.12909 | 0.11482 | 0.11513 | 0.12635 | 0.14581 | 0.09851 | 0.16532 | 0.12369 | lower separation | prior mechanism evidence; not rerun | not recorded | not recorded |
| 6 | R1_RELATIONAL | 0.14374 | 0.14742 | 0.12533 | 0.11355 | 0.14778 | 0.18093 | 0.08319 | 0.22283 | 0.14475 | lower separation | prior mechanism evidence; not rerun | not recorded | not recorded |
| 7 | B0_ACT | 0.16846 | 0.16428 | 0.18935 | 0.15333 | 0.16326 | 0.17624 | 0.14712 | 0.19600 | 0.17037 | lower separation | prior mechanism evidence; not rerun | not recorded | not recorded |
| 8 | R3_FILM | 0.12688 | 0.13042 | 0.10918 | 0.12569 | 0.12689 | 0.13867 | 0.09638 | 0.17462 | 0.13765 | higher separation | prior mechanism evidence; not rerun | not recorded | not recorded |
| 9 | R2_ACTION_LEVEL | 0.13521 | 0.13466 | 0.13792 | 0.12488 | 0.13284 | 0.14627 | 0.11475 | 0.16295 | 0.14716 | lower separation | prior mechanism evidence; not rerun | not recorded | not recorded |

## Interpretation
- All group metrics use the frozen P0.1 x/y tertiles and Edge top-25% definition; boundaries were fixed before comparing models.
- Later/full chunk metrics are prediction-distribution diagnostics because Formal4 exposes a single teacher-forced action target per sample; they are not full-chunk ground-truth errors.
- The current first-action outputs do vary by position, but separation is modest relative to within-group action variance. This is a center-bias diagnostic, not a pass/fail gate.
- Matched-state validation is underpowered: only 2 independent Dev pairs were available from the previous H0 audit.

## Shortlist
1. `B3_ACT`: ACT baseline required for real-robot comparison.
2. `P1_OCACT`: best object-aware candidate by overall h0 action error among this screen.
3. `R3_FILM`: structurally different object-conditioned candidate retained because it has the strongest Left/Right first-action separation diagnostic, despite weaker overall error.

## Boundary
This result authorizes only creation of the next G1 real-robot decision by the user + ChatGPT. It does not authorize robot execution, Final Test, retraining, or architecture changes.
