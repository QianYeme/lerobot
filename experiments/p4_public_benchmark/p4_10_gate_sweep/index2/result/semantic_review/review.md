# Retrieved rollout evidence — SEMANTIC_REVIEW_INCOMPLETE

Locked cases from r0 paired table: Gate+.05-only successes42,43,48; ACT-only success38; first shared failure35. Actual videos and full state/action trajectories for BOTH policies are packaged, not just server paths. Twelve equispaced video-frame samples per clip generated10 contact sheets; only Gate+.05 init42 and38 sheets were visually inspected in this audit. This is not continuous playback, independent human GT, or precise contact annotation.

Init42 sampled views show approach to the blue can, raised/transported can and basket approach. Init38 sampled views show lift/transport-like arm motion but the can visible outside the basket near the end. Contact establishment, slip timing and first failure frame remain uncertain from these sparse views. Do not infer DINO target tracking from arm motion.

Frame-matched on-policy DINO maps/overlays were not newly reconstructed. Existing r0 Train/Dev maps do not establish tracking during these rollouts. Therefore stage-by-stage target/background/robot attention audit is INCOMPLETE. No positive semantic gate or model superiority claim is made.
