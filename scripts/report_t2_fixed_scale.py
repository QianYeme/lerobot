#!/usr/bin/env python
"""Aggregate T2 fixed-scale Dev evaluations without touching Final Test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(root: Path, model: str) -> list[dict]:
    out = []
    for p in sorted((root / "dev_eval" / model).glob("*/metrics.json"), key=lambda x: int(x.parent.name)):
        d = json.loads(p.read_text(encoding="utf-8"))
        out.append({"step": int(p.parent.name), "checkpoint": d["checkpoint"], "metrics": d["metrics"], "fixed_scale": d.get("fixed_scale")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--result-root", required=True); args = ap.parse_args()
    root = Path(args.result_root); data = {m: rows(root, m) for m in ("B2", "P3")}
    best = {m: min(v, key=lambda r: r["metrics"]["S0_normal"]["all"]) for m, v in data.items() if v}
    if set(best) != {"B2", "P3"}:
        raise RuntimeError("Both B2 and P3 fixed-Dev evaluations are required")
    p3 = best["P3"]["metrics"]; base = p3["S0_normal"]["all"]
    ablation = {k: abs(p3[k]["all"] - base) for k in ("S1_zero", "S2_shuffle", "S3_x_only", "S4_y_only")}
    transplant_path = root / "transplant_P3" / "metrics.json"
    transplant = json.loads(transplant_path.read_text(encoding="utf-8")) if transplant_path.exists() else None
    checks = {"horizontal": None, "longitudinal": None}
    if transplant:
        whole = transplant.get("whole", {})
        checks["horizontal"] = whole.get("horizontal", {}).get("horizontal_direction", {}).get("rate")
        checks["longitudinal"] = whole.get("longitudinal", {}).get("longitudinal_direction", {}).get("rate")
    mechanism_status = "pending_transplant"
    if transplant:
        mechanism_status = "PASS" if all(v is not None and v >= 0.70 for v in checks.values()) and max(ablation.values()) > 1e-6 else "FAIL"
    payload = {"experiment": "T2_FIXED_SCALE_SPATIAL_CONDITIONING", "models": data, "best": best, "p3_spatial_ablation_abs_dev_l1": ablation, "transplant": transplant, "mechanism_status": mechanism_status, "criteria": {"directional_consistency_min": 0.70, "path_action_sensitivity_min": 1e-6}, "final_test_used": False, "real_robot_used": False}
    (root / "checkpoints.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    b2 = best["B2"]["metrics"]["S0_normal"]["all"]; p3v = best["P3"]["metrics"]["S0_normal"]["all"]
    text = "# T2 Fixed-Scale Spatial Conditioning\n\nDev-only Formal4 evaluation; Final Test and real robot were not used.\n\n"
    text += f"- B2 best step `{best['B2']['step']}` S0 first-action L1: `{b2:.6f}`.\n- P3 best step `{best['P3']['step']}` S0 first-action L1: `{p3v:.6f}`.\n- P3 fixed residual scale: `0.5` (non-trainable buffer).\n- P3 S0–S4 maximum absolute Dev-L1 change: `{max(ablation.values()):.6g}`.\n"
    if transplant:
        text += f"- Token transplant whole horizontal consistency: `{checks['horizontal']}`; longitudinal: `{checks['longitudinal']}`.\n- Mechanism gate: **{mechanism_status}**.\n"
    else:
        text += "- Token transplant: pending; no mechanism PASS/FAIL is claimed yet.\n"
    text += "\nNo Final Test, real-robot test, or confirmation campaign is authorized by this report.\n"
    (root / "summary.md").write_text(text, encoding="utf-8")
    (root / "mechanism.json").write_text(json.dumps({"status": mechanism_status, "p3_ablation": ablation, "transplant": transplant, "final_test_used": False, "real_robot_used": False}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
