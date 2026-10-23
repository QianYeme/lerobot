# R4 execution result

Outcome: REPRODUCIBILITY_FIXED_RUNTIME. R4_REPRODUCIBILITY_PASS=true.
R4_EXECUTION_COMPLETE=true; R4_COMPLETE=false pending independent external Builder/Red-Team acceptance.

Same-process frozen-input outputs were identical. Historical cross-process outputs diverged first at the observed backbone hook. Only cudnn.benchmark changed True to False; matmul/cudnn TF32 stayed enabled. Fixed A/B/C and full D on seed3/seed1 passed the original <0.001 gate. Specific cuDNN kernel IDs were not captured; do not claim operator-level localization.

All critical globals were freshly replayed, frozen locals recomposed on fresh globals, and the original grouping/matching/classification code reused. See postfix_f5_1_metrics.csv and postfix_f5_1/ for numerical results and conflicts, not pre-fix numbers. All Full60 findings remain TRAIN-SET / COMMON-DIAGNOSTIC ONLY, NOT HELD-OUT GENERALIZATION. No training, data collection, real robot, actual deployment configuration changes or publication.
