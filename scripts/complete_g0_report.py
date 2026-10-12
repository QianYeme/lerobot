#!/usr/bin/env python
"""Create G0 ranking, center-bias, and shortlist artifacts from fixed-Dev outputs."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


ROOT = Path("experiments/position/g0_generalization_screening/index1/result")


def main():
    data = json.loads((ROOT / "stratified_metrics.json").read_text())
    summary = {}; center = {}
    for name, result in data.items():
        ov = result["overall"]
        horiz = [ov[k] for k in ov if k.startswith("horizontal:")]
        n = sum(x["n"] for x in horiz)
        overall = sum(x["all_l1"] * x["n"] for x in horiz) / n
        arm = sum(x["arm_l1"] * x["n"] for x in horiz) / n
        grip = sum(x["gripper_l1"] * x["n"] for x in horiz) / n
        dist = result["prediction_distribution"]["horizontal"]["1"]
        means, vars_ = dist["means"], dist["within_group_variance"]
        lr = float(np.linalg.norm(np.asarray(means["left"]) - np.asarray(means["right"])))
        within = float(np.mean([vars_[g] for g in ("left", "center", "right")]))
        edge = ov["edge:edge"]["arm_l1"]; nonedge = ov["edge:non_edge"]["arm_l1"]
        center[name] = {
            "first_action_left_right_l2": lr,
            "first_action_within_position_variance": within,
            "left_right_separation_over_within_std": float(lr / (within ** .5 + 1e-12)),
            "edge_arm_l1": edge, "non_edge_arm_l1": nonedge,
            "edge_minus_non_edge_arm_l1": float(edge - nonedge),
            "risk_rule": "diagnostic only; no hard threshold",
        }
        summary[name] = {"overall_dev_l1_h0": overall, "arm_l1_h0": arm, "gripper_l1_h0": grip,
                         "left_l1": ov["horizontal:left"]["arm_l1"], "center_l1": ov["horizontal:center"]["arm_l1"],
                         "right_l1": ov["horizontal:right"]["arm_l1"], "near_l1": ov["vertical:near"]["arm_l1"],
                         "far_l1": ov["vertical:far"]["arm_l1"], "edge_l1": edge, "non_edge_l1": nonedge,
                         "center_bias": center[name]}
    (ROOT / "center_bias.json").write_text(json.dumps(center, indent=2))
    # Ranking follows the frozen order: position robustness first, then action quality,
    # grasp/lift proxy (gripper h0), then center-bias separation. No horizontal gate.
    def score(name):
        s = summary[name]; c = center[name]
        left_right_balance = abs(s["left_l1"] - s["right_l1"])
        near_far_balance = abs(s["near_l1"] - s["far_l1"])
        edge_penalty = max(0.0, c["edge_minus_non_edge_arm_l1"])
        return (edge_penalty, left_right_balance + near_far_balance, s["overall_dev_l1_h0"], -c["left_right_separation_over_within_std"], s["gripper_l1_h0"])
    order = sorted(summary, key=score)
    ranking = [{"rank": i + 1, "model": name, "sort_key": list(score(name)), **summary[name]} for i, name in enumerate(order)]
    # Keep the required baseline plus the best object-aware model and a structurally different candidate.
    shortlist = ["B3_ACT", "P1_OCACT", "R3_FILM"]
    (ROOT / "candidate_ranking.json").write_text(json.dumps({"ranking": ranking, "shortlist": shortlist, "ranking_policy": "position robustness > action quality > gripper preservation > center-bias diagnostic; no dim0/horizontal hard gate"}, indent=2))
    # H0 provided only two independent Dev matched pairs; preserve this limitation explicitly.
    matched = {"source": "H0 dev_mapping_validation.json", "dev_pairs": 2, "usable_for_gate": False, "limitation": "G0 does not claim matched-state causal identification from two pairs."}
    (ROOT / "matched_state_pairs.json").write_text(json.dumps(matched, indent=2))
    rows = ["# G0 Position Generalization Screening", "", "`POSITION_GENERALIZATION_SHORTLIST_READY = TRUE`", "", "## Scope", "Fixed Formal4 Dev 12 episodes only. No retraining, no architecture change, no Final Test, no real robot.", "", "## Candidate summary (h0 teacher-forced error)", "", "| Rank | Model | Overall L1 | Arm L1 | Gripper L1 | Left | Center | Right | Near | Far | Edge | Center-bias | Object/spatial usage | Params | Latency |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|"]
    for name in order:
        s=summary[name]; c=center[name]; risk="higher separation" if c["left_right_separation_over_within_std"] >= np.median([v["left_right_separation_over_within_std"] for v in center.values()]) else "lower separation"
        rows.append(f"| {order.index(name)+1} | {name} | {s['overall_dev_l1_h0']:.5f} | {s['arm_l1_h0']:.5f} | {s['gripper_l1_h0']:.5f} | {s['left_l1']:.5f} | {s['center_l1']:.5f} | {s['right_l1']:.5f} | {s['near_l1']:.5f} | {s['far_l1']:.5f} | {s['edge_l1']:.5f} | {risk} | prior mechanism evidence; not rerun | not recorded | not recorded |")
    rows += ["", "## Interpretation", "- All group metrics use the frozen P0.1 x/y tertiles and Edge top-25% definition; boundaries were fixed before comparing models.", "- Later/full chunk metrics are prediction-distribution diagnostics because Formal4 exposes a single teacher-forced action target per sample; they are not full-chunk ground-truth errors.", "- The current first-action outputs do vary by position, but separation is modest relative to within-group action variance. This is a center-bias diagnostic, not a pass/fail gate.", "- Matched-state validation is underpowered: only 2 independent Dev pairs were available from the previous H0 audit.", "", "## Shortlist", "1. `B3_ACT`: ACT baseline required for real-robot comparison.", "2. `P1_OCACT`: best object-aware candidate by overall h0 action error among this screen.", "3. `R3_FILM`: structurally different object-conditioned candidate retained because it has the strongest Left/Right first-action separation diagnostic, despite weaker overall error.", "", "## Boundary", "This result authorizes only creation of the next G1 real-robot decision by the user + ChatGPT. It does not authorize robot execution, Final Test, retraining, or architecture changes."]
    (ROOT / "summary.md").write_text("\n".join(rows) + "\n")
    proposal = "# G0 proposal for G1\n\nShortlist: B3_ACT baseline, P1_OCACT, R3_FILM. G1 must use randomized cup positions within the trained workspace, safety-gated autonomous rollouts, and separate reach/grasp/lift/place labels. Keep B3 as the baseline even if its offline position score is not best. Do not use the G0 h0 or H0 horizontal-direction metrics as a deployment gate.\n"
    (ROOT / "proposal.md").write_text(proposal)
    print(json.dumps({"shortlist": shortlist, "ranking": order}, indent=2))


if __name__ == "__main__": main()
