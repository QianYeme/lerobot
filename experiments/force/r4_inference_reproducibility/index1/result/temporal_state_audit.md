# Temporal/cache audit

Frozen checkpoint temporal_ensemble_coeff=null. Evaluator calls predict_action_chunk directly, not select_action; no temporal ensemble, action-queue consumption, or observation history is used. eval disables VAE sampling and dropout. Probe logs queue before/after; full replay resets every batch and logs empty action queue. Per-batch episode/frame manifests preserve episode boundaries; stateless evaluation does not require sequential carryover. This is NOT an audit of the actual robot temporal-ensemble controller.
