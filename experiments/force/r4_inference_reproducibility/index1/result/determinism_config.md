# Runtime control

Historical: cudnn.benchmark=True, cudnn.deterministic=False, deterministic_algorithms=False, matmul/cudnn TF32=True. Final audit: benchmark=False only; TF32 remains True; no new training seeds or precision/model changes. Original RNG hashes, library versions, dropout/eval flags and cross-process configs are in audit_config.json. Worker seeds use the DataLoader default torch-seed+worker-id policy. No random replay augmentation was configured. Passed on this server/version/checkpoint set; portability to another GPU/runtime is not established.
