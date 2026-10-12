#!/usr/bin/env python
"""Aggregate T3 joint state-spatial Dev evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(root: Path, model: str) -> list[dict]:
    rows = []
    for p in sorted((root / "dev_eval" / model).glob("*/metrics.json"), key=lambda x: int(x.parent.name)):
        d = json.loads(p.read_text(encoding="utf-8")); rows.append({"step": int(p.parent.name), "checkpoint": d["checkpoint"], "metrics": d["metrics"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--result-root", required=True); args = ap.parse_args()
    root = Path(args.result_root); all_rows = {m: load_rows(root, m) for m in ("B3", "P4")}
    best = {m: min(v, key=lambda r: r["metrics"]["J0_normal"]["all"]) for m, v in all_rows.items() if v}
    if set(best) != {"B3", "P4"}: raise RuntimeError("B3 and P4 Dev evaluations are required")
    p4 = best["P4"]["metrics"]; base = p4["J0_normal"]["all"]
    counterfactual = {k: abs(p4[k]["all"] - base) for k in ("J1_spatial_zero", "J2_xy_shuffle", "J3_x_transplant", "J4_y_transplant", "J5_full_spatial_transplant")}
    transplant_path = root / "transplant_P4" / "metrics.json"; transplant = json.loads(transplant_path.read_text()) if transplant_path.exists() else None
    horizontal = longitudinal = None
    if transplant:
        horizontal = transplant["whole"]["horizontal"]["horizontal_direction"]["rate"]
        longitudinal = transplant["whole"]["longitudinal"]["longitudinal_direction"]["rate"]
    mechanism = "pending_transplant"
    if transplant:
        mechanism = "PASS" if horizontal >= .70 and longitudinal >= .70 and max(counterfactual.values()) > 1e-6 else "FAIL"
    payload = {"experiment": "T3_JOINT_STATE_SPATIAL_CONDITIONING", "models": all_rows, "best": best, "joint_counterfactual": counterfactual, "transplant": transplant, "mechanism_status": mechanism, "criteria": {"horizontal": .70, "longitudinal": .70, "final_test_used": False, "real_robot_used": False}}
    (root / "checkpoints.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    text = "# T3 Joint State-Spatial Conditioning\n\nDev-only Formal4 evaluation; Final Test and real robot were not used.\n\n"
    text += f"- B3 best step `{best['B3']['step']}` J0 first-action L1: `{best['B3']['metrics']['J0_normal']['all']:.6f}`.\n- P4 best step `{best['P4']['step']}` J0 first-action L1: `{best['P4']['metrics']['J0_normal']['all']:.6f}`.\n- P4 joint counterfactual maximum Dev-L1 change: `{max(counterfactual.values()):.6g}`.\n"
    text += f"- Joint counterfactual changes: `{json.dumps(counterfactual, sort_keys=True)}`.\n"
    if transplant: text += f"- Whole-object transplant horizontal consistency: `{horizontal:.3f}`; longitudinal: `{longitudinal:.3f}`.\n- Mechanism gate: **{mechanism}**.\n"
    else: text += "- Token transplant is pending; no mechanism PASS/FAIL is claimed yet.\n"
    text += "\nNo Final Test, real-robot test, or confirmation campaign is authorized by this report.\n"
    (root / "summary.md").write_text(text, encoding="utf-8")
    (root / "mechanism.json").write_text(json.dumps({"status": mechanism, "counterfactual": counterfactual, "transplant": transplant, "final_test_used": False, "real_robot_used": False}, indent=2), encoding="utf-8")
    (root / "resources.json").write_text(json.dumps({"final_test_used": False, "real_robot_used": False, "models_evaluated": list(all_rows), "checkpoint_selection": "lowest Dev J0 L1"}, indent=2), encoding="utf-8")
    (root / "proposal.md").write_text("# T3 follow-up proposal\n\nIf T3 fails, stop absolute spatial conditioning and design a separate relative target-state representation decision.\n", encoding="utf-8")


if __name__ == "__main__": main()
