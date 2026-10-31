# Confirmation-init capacity blocker

Decision B requires freezing unused confirmation IDs, recommends >=50 paired rollouts/model, and explicitly says to submit a blocker/alternative if environment capacity does not support it. Section7 says: 如阶段 A/B 出现阻断或证据相互矛盾，生成 blocking_proposal.md，保留已完成结果停止推进。

Verified official task init bank has exactly50 states (prior_init_contract.json, legal_init_state_count=50). IDs10–29 were used in r5/P4.9;30–49 in r0. Thus at most10 IDs remain,0–9, with earlier smoke usage uncertain. No set of50 unused legal IDs can be allocated. Changing reset seeds does not create new official init states; repeat rollouts are not independent starts.

Stop before new training: A3 fixed-alpha5k, B new seeds/Legacy, C formal6 pilots, final80k and robot are NOT_STARTED. This does not invalidate r0 exploratory results or prove Object Path ineffective.

Alternative requiring a revised decision: (1) audit0–9 history, use only proven-unused IDs as a small confirmation set, explicitly reduce independent-start count and report low power; or (2) separately approve a new task or a preregistered physically valid reset-distribution generator, validate legality and freeze a disjoint50-ID set before selection. Do not silently jitter official starts or invent a second seed. If only task/seed replication is allowed, label it replication on known starts, not independent unseen-state confirmation.

No confirmation model outcomes have been inspected; no new gate or prototype was selected. Recommend NO_GATE_CHANGE / decision deferred for both L1/F3 until the blocking design is revised.
