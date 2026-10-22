# F5 deployment preparation

- F5_COMPLETE: false
- Formal robot trials: 0 / 12; no autonomous success rate is available.
- Four candidates: P1-Hist, Strong-G, L-Load, New-GL, as frozen by F5 r0.
- Artifacts: https://huggingface.co/QYyyyyyyy/F5_UNIFIED_SO101
- Source branch: https://github.com/QianYeme/lerobot/tree/det

This package delivers the unified controller, exact selected global/local weights,
normalization, cup prototype, offline vision assets, four independent commands,
trial identity/video/control logging and reviewed-stage summary tooling.
Temporal ensemble and robot relative target limit remain CLI-adjustable; defaults
are 0.01 and 5, not mandatory values. Freeze common settings before formal trials.

Validation is limited to 20 local mock/CLI regressions and actual Dev first-frame
GPU smoke tests on the server. No real robot was connected; no new training or
Final12 access was performed. The first/reset-step test is not a multistep
closed-loop validation or a safety guarantee.

The current ACT file differs from the early F2.1 training snapshot by an optional
load gate branch. Do not describe the source files as AST-identical. Model hashes
and inference route checks are separate evidence from source identity.

See hf_publish_receipt.json for the verified publication revision/file hashes
once publication has completed, and REAL_ROBOT_TEST_GUIDE.md in the parent folder
for setup, trial order and commands. The only next experiment is the user-operated
12-trial F5 protocol, not automatic additional training or model selection.
