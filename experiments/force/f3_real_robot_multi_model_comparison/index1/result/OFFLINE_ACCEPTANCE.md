# F3 r1 offline engineering acceptance

2026-10-20; westC:30824; RTX5090; PyTorch2.8.0+cu128. No robot connection, training, or Final12 access.

## Completed

- Original checkpoint and fixture retained. Four CPU/CUDA precision modes repeated three times; original batch16 reference passes under its original TF32 mode (3.576e-7), without changing tolerance.
- Original LocalResidual class extracted from archived F2.3 training source via AST, strictly loaded with the same checkpoint. Adapter-versus-original errors are exactly zero for batch16 and individual batch1 calls in all four modes. Evidence: precision_batch1_audit.json, including source hash.
- CUDA TF32 batch1-versus-batch16 difference is 0.01770389; full-precision differences are below 1.5e-6. This is batch-dependent numerical behavior, not evidence of different weights. Do not compare batch1 against a batch16 TF32 expected array. No reference was replaced.
- Four-model unified inference compared against one real Dev observation: all preprocessed features and nominal native actions exactly match reference pipelines. M1/M3 disabled agree; shadow preserves commands; regulated preserves arm columns and bound; disabled needs no local load/current. Evidence: unified_runtime_smoke.json.
- Fake-robot integration verifies CSV/video, reset/disconnect, and injected inference failure before sending an action, with persistent M3 stop marker. Evidence: unified_runtime_mock.json. Its ERROR is deliberately injected, not a robot trial.
- Server regression: 12 tests PASS (final_pytest.log). Pytest installed solely to run these checks. No model/data/scientific configuration changed.

## Delivery and limits

Unified entry: src/lerobot/scripts/control_f3.py; routing: f3_runtime.py; protocol: f3_control_protocol.py. Commands end the guide at ../REAL_ROBOT_TEST_GUIDE.md. Approval templates default false. The runner records its common TF32 setting in the session fingerprint; this does not authorize changing precision after observing outcomes. Target hardware must run offline checks and freeze common settings before Stage0.

G1 provenance: clip=5 and no explicit ensemble override, not clip=2/ensemble=0.01. Neither candidate is a hardware safety certification. Original HF bundle remains frozen and lacks the new unified runner; use updated Git source for that entry.

Server evidence: /root/autodl-tmp/F3_control_validation; untouched bundle: /root/autodl-tmp/F3_r1_transfer_20261020/bundle. Large files stay on server, not Git.

Physical calibration, camera/port checks, emergency stop, real timing, shadow3, 12 formal trials, behavior scoring and scientific Builder/Red-Team reviews remain NOT_RUN. F3_R1_COMPLETE=false. Engineering acceptance is not grasp success or force-control evidence.
