# F3 M3 numerical reproducibility audit — TF32 dependence reproduced

Date: 2026-10-20. Engineering diagnosis only; no robot, training, or Final12 access.

## Follow-up: new server verification completed

The new user-supplied server westC:30824 contains the unchanged bundle. PyTorch 2.8.0+cu128 / CUDA 12.8 / RTX 5090. `scripts/audit_f3_precision.py` imports the bundled adapter directly and evaluates the original 16 fixture rows, three repeats per mode. Full outputs and artifact SHA256 are preserved in `precision_audit_20261020.json` locally and `/root/autodl-tmp/F3_control_validation/precision_audit_20261020.json` on the server.

| Mode | Maximum reference error | Original 1e-4 threshold |
|---|---:|---|
| CPU FP32 | 0.01770192385 | FAIL |
| CPU FP64 | 0.01770577401 | FAIL |
| CUDA FP32, TF32 off | 0.01770424843 | FAIL |
| CUDA FP32, TF32 on | 0.00000035763 | PASS |

All three repeats within each mode are identical. No checkpoint, input, normalization, expected value, or tolerance was changed. The observed fixture discrepancy is explained by the numerical path; the earlier assertion that it proves a different checkpoint generated the fixture is unsupported and should not be retained as the diagnosis.

Scope: this reproduces the original batched fixture on this GPU, not deployment across GPUs or batch sizes. Independent comparison against the original training implementation, real control batch-size behavior, full bundle validation, and deployment Stage0 remain pending. Do not silently change deployment precision or declare M3 robot-ready. Existing lower sections preserve the prior blocked-state chronology.

## Evidence and interpretation

- Remote commit `2da7c63b` records a local fixture error of about 0.0177 and concludes that the fixture came from different weights. That conclusion is not yet established by the recorded checks.
- The earlier `f3_real_robot_local_regulator_feasibility/index1/result/deployment_validation.md` explicitly records approximately 0.0177 under default numerical settings, then 3.5762787e-7 with the original CUDA/TF32 settings. `scripts/verify_f3_bundle.py` sets `torch.backends.cuda.matmul.allow_tf32=True`.
- Agreement between CPU FP32 and FP64 does not rule out disagreement with a CUDA TF32-generated reference. Likewise, matching file hashes establishes file identity, not numerical portability. These are competing explanations, not proof of a mismatched checkpoint.
- The latest remote log reports 639 non-pyc payload hashes matching. This was not independently rerun in this audit. Mutable Python bytecode should be excluded from future publication manifests; the current manifest and original artifacts must remain intact as evidence.

## Required discriminating check

Use the same checkpoint, fixture inputs, normalization, and adapter source. Record their hashes and Python/PyTorch/GPU versions. Run CPU FP32 and FP64, CUDA FP32 with TF32 off, and CUDA with the original TF32 setting. Record every output and maximum error against the unchanged reference. Repeat runs to establish reproducibility. Separately compare the adapter against the original F2.3 local implementation under identical numerical settings.

Do not replace expected values, substitute checkpoints, or widen the 1e-4 tolerance to obtain PASS. If the reference depends on numerical mode, document that dependency and validate the intended deployment mode before unblocking M3.

## Deployment parameter source audit

The checked-in G1 guide's three inference commands use `--robot.max_relative_target=5`; they do not explicitly override temporal ensembling. The four exported checkpoint configs were previously recorded as `temporal_ensemble_coeff=null`. Therefore the new log's claim that clip=2 and ensemble=0.01 directly inherit those G1 commands is not supported by that source.

Treat 5/null as provenance evidence, not an automatic safety approval. The F3 common deployment configuration remains PENDING until explicitly frozen for the actual hardware. Do not silently choose either parameter set. Unified-entry changes currently in the local worktree remain uncommitted and are not a published deployment release.

## Actual execution and blocker

The SSH attempt to the last supplied server, `connect.weste.seetacloud.com:53289`, was refused. The current local workspace has no `outputs/deploy/F3_R1` bundle and its available project Python has no PyTorch. Thus the numerical matrix was NOT_RUN; no root cause or numerical PASS is claimed.

Git fetch obtained `origin/det` commit `2da7c63b`; fast-forward merge was rejected because local STATUS and work-log edits would be overwritten. Those edits were preserved. Remote content was inspected using `git show`, without resetting or stashing the worktree.

Next dependency: an accessible server containing the unchanged F3 bundle, or a deployment host with the bundle and PyTorch. M3 shadow/regulated and all real-robot stages remain blocked pending verification and on-site safety gates.
