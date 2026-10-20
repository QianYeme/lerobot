# Implementation difference

Selected full configs differ ONLY in load_feature_gating=false/true. Full common initialization equal (gpu_smoke.json). Gate is multiplied after load MLP and does not gate state/image/action or remove positional embedding.

implementation_patch.diff compares against preserved F2.1 source; hashes pin actual runtime files. Old runner reused unchanged; subclass overrides config, batch and diagnostics. New evaluation never trains.
