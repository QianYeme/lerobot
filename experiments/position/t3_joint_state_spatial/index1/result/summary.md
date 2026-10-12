# T3 Joint State-Spatial Conditioning

Dev-only Formal4 evaluation; Final Test and real robot were not used.

- B3 best step `100000` J0 first-action L1: `0.085466`.
- P4 best step `100000` J0 first-action L1: `0.126716`.
- P4 joint counterfactual maximum Dev-L1 change: `3.7116e-05`.
- Joint counterfactual changes: `{"J1_spatial_zero": 3.711595514332977e-05, "J2_xy_shuffle": 1.270644154172107e-06, "J3_x_transplant": 1.0526770817809172e-06, "J4_y_transplant": 2.1969231989160676e-07, "J5_full_spatial_transplant": 1.270644154172107e-06}`.
- Whole-object transplant horizontal consistency: `0.000`; longitudinal: `0.000`.
- Mechanism gate: **FAIL**.

No Final Test, real-robot test, or confirmation campaign is authorized by this report.
