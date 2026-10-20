# F2.2 unattended handoff

Current state: RUNNING, not yet a completed result.

Server root: `/root/autodl-tmp/F2_2_r0_20261019`

Screens: `F22_B0`, `F22_B1`, `F22_AUTOFINISH`.
Training processes/source/configuration were not changed or restarted.

1. Wait for both formal100k runs and exit0; start only a formal variant that has never started.
2. Audit all fixed20k checkpoint evaluations and earliest-minimum primary selection.
3. Strict-load selected weights and evaluate frozen Dev5748 frames, per-episode physical/normalized errors.
4. Gate-on/off, encoded-feature-zero and separate normalized-input-zero; save raw predictions.
5. Gate activities, health, implementation difference, input/source identities, leakage audit.
6. Idle-GPU batch1 inference benchmark, eight figures, complete result and CRC-verified ZIP.
7. Write READY and stop. Wait for independent user/planner Builder and Red-Team review.

Server state: `autofinish_state.json`; errors: timestamped `autofinish_failure_*.log`
and `delivery/proposal_failure_*.md`. Existing failed training is never silently retrained.
No scientific changes, new experiments, extra budget, Final, robot or publication.

Final artifacts: `delivery/`, `READY.json`, `F2_2_result.zip`.
All checkpoints/logs/intermediate evidence remain on server; ZIP includes checkpoint hashes,
not multi-GB model files. Figure raster validation is automated; external visual review remains.

Optional local receiver is already running hidden for24h. With this computer online it
downloads and validates the ZIP, archives pretraining reports, and extracts to this result.
Its state is `AUTO_DELIVERY_STATE.json`, log `outputs/f2_2_receiver.log`.
Closing SSH or this conversation does not stop server execution. Turning off the local
computer affects only automatic reception; the server package remains available.

Verified engineering checks: real-GPU delivery smoke; six local regression tests including
synthetic TEMP-only end-to-end report/figure/ZIP fixture. Synthetic data is NOT research evidence.
