#!/usr/bin/env python
"""Prepare the frozen G1 real-robot trial order without touching hardware."""
from __future__ import annotations

import json
import random
from pathlib import Path


OUT = Path("experiments/position/g1_real_robot_generalization/index1/result")
MODELS = ["B3_ACT", "P1_OCACT", "R3_FILM"]
BLOCKS = [["B3_ACT", "P1_OCACT", "R3_FILM"], ["R3_FILM", "B3_ACT", "P1_OCACT"], ["P1_OCACT", "R3_FILM", "B3_ACT"]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261013)
    rows = []
    trial_id = 0
    # Phase A: nominal-center function/safety check; excluded from primary rate.
    for model in MODELS:
        for repeat in range(1, 4):
            trial_id += 1
            rows.append({"trial_id": f"A{trial_id:03d}", "phase": "A_safety_function", "model": model, "repeat": repeat, "stratum": "nominal_center", "primary_metric_eligible": False, "intended_x_bin": 1, "intended_y_bin": 1})
    # Phase B: 3 randomized model blocks, 12 strata x 2 repeats/model.
    for block_id, order in enumerate(BLOCKS, 1):
        strata = [(xb, yb) for xb in range(4) for yb in range(3)]
        rng.shuffle(strata)
        block_strata = strata[(block_id - 1) * 4 : block_id * 4]
        for model in order:
            for repeat in range(1, 3):
                for xb, yb in block_strata:
                    trial_id += 1
                    rows.append({"trial_id": f"B{trial_id:03d}", "phase": "B_generalization", "block": block_id, "model": model, "repeat": repeat, "stratum": f"x{xb}_y{yb}", "primary_metric_eligible": True, "intended_x_bin": xb, "intended_y_bin": yb, "realized_x_mm": None, "realized_y_mm": None})
    payload = {"experiment": "G1_POSITION_REAL_ROBOT_GENERALIZATION", "seed": 20261013, "models": MODELS, "phase_a_trials": 9, "phase_b_trials": 72, "total_trials": len(rows), "strata": {"horizontal_bins": 4, "longitudinal_bins": 3, "workspace_mm": {"x_total": 40, "y_total": 30}}, "final_test_used": False, "trial_order_frozen": True, "hardware_preflight": {"robot_serial": "NOT_DETECTED", "cameras": "NOT_VERIFIED", "checkpoints_local": "NOT_PRESENT", "status": "BLOCKED_HARDWARE"}, "trials": rows}
    (OUT / "trial_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT / "selected_candidate.json").write_text(json.dumps({"models": MODELS, "selection_source": "G0 frozen shortlist", "status": "awaiting_hardware_and_operator_confirmation", "checkpoint_refs": {m: "G0 frozen 100k checkpoint; local deployment path not supplied" for m in MODELS}}, indent=2), encoding="utf-8")
    (OUT / "preflight.md").write_text("""# G1 deployment preflight\n\nStatus: `BLOCKED_HARDWARE`\n\nThe current Codex workstation has the required NVIDIA RTX 4060 Laptop GPU, but no SO101 serial port/cameras were detected and the frozen B3/P1/R3 checkpoints are not present locally. No real-robot trial was started.\n\nBefore Phase A:\n\n1. Connect the follower arm and verify the actual serial port.\n2. Verify top and wrist camera indices/resolution/FPS and save one frame from each.\n3. Place the robot at the frozen Home + Open pose and verify the safety workspace.\n4. Copy the three frozen G0 checkpoints without changing preprocessing, state normalization, action postprocessing, `n_action_steps`, or safety limits.\n5. Run exactly three nominal-center trials per model, then stop for inspection if any safety abort occurs.\n6. Only after Phase A is safe, execute the frozen Phase-B manifest with the same randomized order for all models.\n\nThe operator must fill realized cup coordinates and phase labels immediately after each trial; labels must not be reconstructed from memory after seeing all model results.\n""", encoding="utf-8")
    print(json.dumps({"total_trials": len(rows), "phase_a": 9, "phase_b": 72, "status": "BLOCKED_HARDWARE"}, indent=2))


if __name__ == "__main__": main()
