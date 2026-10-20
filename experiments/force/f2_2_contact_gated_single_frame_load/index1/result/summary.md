# F2.2 r0 completed — GATING_HURTS

B0 training DONE; B1 training DONE; Dev12 evaluation DONE; Mechanism diagnostics DONE;
Gate audit DONE; Latency/params DONE; Figures DONE; Result package READY.

B0 selected 100000 steps: normalized h0 gripper L1 0.13219491,
denormalized gripper MAE 3.292596, arm MAE 3.194060.
B1 selected 40000 steps: normalized h0 gripper L1 0.18290253,
denormalized gripper MAE 4.555578, arm MAE 4.064688.
B1 primary relative change +38.36%; improves 0/12 episodes.
Dev gate active 50.92%; threshold unchanged 44.146005630493164; latency ratio 1.052.

Single-seed exploratory offline evidence only. Gate is a position-based grasp proxy, not contact ground truth.
Dev selection/reporting reuse creates optimism; no Final or real robot. No actual force-control effectiveness claim.
Unfavorable evidence is retained. All10 checkpoint weights and logs remain on server, inventory with hashes attached.
ZIP contains reports/raw evaluations/logs/predictions/source, not multi-GB weights or videos.

Execution stops here. Independent Builder Review and Red-Team Review await user/planner.
