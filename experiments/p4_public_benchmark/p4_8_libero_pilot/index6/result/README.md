# P4.8-r6 diagnostic delivery

Start with summary.md and mechanism_findings.md. This package contains diagnostics only, no checkpoint weights.

Sources: index6/decision.md; result/upstream_manifest.json points to the six existing r5 checkpoints and frozen source/config hashes. Input lock contains Dev27+Train9+four bank-reference frames; no Final. Every numeric table declares model, checkpoint, probe, units, location and eval dtype.

Reproduction source is under result/tools. The executed copies are at index6/workspace on the server, using /root/autodl-tmp/p48_isolated_env/bin/python and the frozen r5 workspace/scripts factory. prepare verifies upstream hashes; probe --kind legacy/fixed runs D0-D4, with no model save/update; render_summary produces D3 heatmaps and D5 fixed video samples; parameter_deltas reads5k/10k states; complete_chunk_metrics supplies per-dimension normalized discrete indices and arm/gripper chunk maxima, asserting reproduction against original dose rows. Original per-model/raw tables and logs remain alongside merged detailed tables.

D2 activates native CVAE posterior loss but disables Dropout including MHA, fixes latent seed, and never advances an optimizer. This is explicitly different from eval-prior action prediction. Initial full hash is checked before calling initial gradients.

A shuffled prototype-slot bank is mathematically invariant under max matching; random non-ROI Train patches are only a control, not guaranteed semantically unrelated features. Temporary gate/input/bank interventions are diagnostic/OOD and are never sent to an environment. All states/checkpoint files remain intact.

The existing r5 videos are represented by sampled overview images and SHA/path pointers, not re-uploaded. R6 adds zero rollout or training updates. The next_experiment_proposal is unapproved and not executed. Independent Builder/Red-Team review remains PENDING.
