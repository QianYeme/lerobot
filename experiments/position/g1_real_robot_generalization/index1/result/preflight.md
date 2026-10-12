# G1 deployment preflight

Status: `BLOCKED_HARDWARE`

The current Codex workstation has the required NVIDIA RTX 4060 Laptop GPU, but no SO101 serial port/cameras were detected and the frozen B3/P1/R3 checkpoints are not present locally. No real-robot trial was started.

Before Phase A:

1. Connect the follower arm and verify the actual serial port.
2. Verify top and wrist camera indices/resolution/FPS and save one frame from each.
3. Place the robot at the frozen Home + Open pose and verify the safety workspace.
4. Copy the three frozen G0 checkpoints without changing preprocessing, state normalization, action postprocessing, `n_action_steps`, or safety limits.
5. Run exactly three nominal-center trials per model, then stop for inspection if any safety abort occurs.
6. Only after Phase A is safe, execute the frozen Phase-B manifest with the same randomized order for all models.

The operator must fill realized cup coordinates and phase labels immediately after each trial; labels must not be reconstructed from memory after seeing all model results.
