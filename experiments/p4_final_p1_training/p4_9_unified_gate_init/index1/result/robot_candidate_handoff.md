# Robot candidate handoff

No real robot was operated. Do not infer physical performance from teacher-forced MAE.
Historical L1 (Legacy seed1) and F3 (Fixed seed3) remain controls. Each new run changes only gate initialization to +0.001; the scalar remains learnable and signed.
Use checkpoint_manifest.csv and model_provenance.json for exact steps, paths and hashes. Only a complete80k formal6 model is eligible for later controlled robot review; PARTIAL is not80k.
Legacy uses the original normalized Top as DINO input (the frozen encoder normalizes it again). Fixed preserves Top [0,1] under observation.images.top_raw for one DINO normalization. Main CNN preprocessing is unchanged. Never interchange these entry paths.
State8 includes six joints plus gripper.load/gripper.curr; action6, chunk100, historical train-only normalizer/prototype bank. Source interface is640x480; nn SO101 calibration/units retain FRESH_TRAIN_LIMITED history and require on-site review. Camera/port history: follower /dev/ttyACM0, leader /dev/ttyACM1, gripper2/top4. This is not authorization to send actions or a new deployment command.
Future pairings: L1 Gate0 vs L1 Gate+ seed1; F3 Gate0 vs F3 Gate+ seed3. Cross-seed Legacy-vs-Fixed differences are confounded. No automatic HF upload, robot test, or next experiment.
