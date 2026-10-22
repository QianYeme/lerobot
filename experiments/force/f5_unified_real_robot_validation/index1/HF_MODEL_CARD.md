---
tags:
- robotics
- lerobot
- so101
- act
---

# F5 unified SO-101 real-robot validation package

Four frozen candidates for the transparent water-filled cup task:

| Model ID | Global | Local regulator |
|---|---|---|
| P1-Hist | Historical P1/M0, state8 including load/current | None |
| Strong-G | F4.1 G-Load seed3, 80k, state6 + single-frame load | None |
| L-Load | Existing no-load state6 global | F4.1 seed4, 9k |
| New-GL | Same global as Strong-G | F4.2 seed3, 7k |

Exactly four test models, stored as three distinct global weights and two local
weights. B0-6D is an internal component of L-Load, not an additional test model.
See model_identity.json for exact component SHA256 values.

## Download and use

Start in the project root from https://github.com/QianYeme/lerobot/tree/det.

```bash
mkdir -p outputs/deploy
hf download QYyyyyyyy/F5_UNIFIED_SO101 F5_deploy_bundle.tar.gz F5_project_files.tar.gz --local-dir outputs/deploy
# Preserve any local changes before unpacking the project overlay.
tar -xzf outputs/deploy/F5_project_files.tar.gz -C .
tar -xzf outputs/deploy/F5_deploy_bundle.tar.gz -C outputs/deploy
echo '9187d7e8db452fa468c3d6930558fb1e6ba9d313bdd52a0a454cf3c6c6f0cca9  outputs/deploy/F5_deploy_bundle.tar.gz' | sha256sum -c -
```

Read REAL_ROBOT_TEST_GUIDE.md before connecting hardware. It ends with four
complete inference commands. Policy ensemble and robot relative-target limit
are CLI-adjustable, defaulting to 0.01 and 5. Freeze common settings across models
before formal comparison. Do not operate a robot from a cloud server.

## Contents and limitations

- F5_deploy_bundle.tar.gz: weights, normalization, cup prototype, offline
  DINO/ResNet assets, identity and payload manifest. Extracts to F5/.
- F5_project_files.tar.gz: minimal F5 source overlay, summary script and guide;
  requires the project and its existing F3 runtime dependencies, not standalone.
- runtime_smoke.json: actual Dev first-frame GPU inference checks only.
- bundle_receipt.json: archive size/hash, created before publication.
- model_identity.json and manifest.json: exact frozen component/payload hashes.

No training dataset or Final12 is included. No F5 robot trials have been run:
F5_COMPLETE=false. Offline checks are not evidence of safe operation, autonomous
success, or calibrated force control. Actuator load is not a force measurement.
Use a physical emergency stop and support the arm before disabling torque.
