#!/usr/bin/env python
"""Re-score box-condition checkpoints with a trajectory-level direction reference.

The original short-run Gate compared three frame-0 actions. This validator keeps
those raw results intact and uses the independent full-trajectory GT slope from
``eval_pan_direction_consistency.py`` as the direction reference instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STEPS = ("004000", "006000", "008000", "010000")
MODES = ("C0", "C1", "C2")


def strictly_monotonic(values: list[float], increasing: bool) -> bool:
    pairs = zip(values, values[1:])
    return all((b > a) if increasing else (b < a) for a, b in pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--direction-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-pan-span", type=float, default=0.05)
    parser.add_argument("--min-box-effect", type=float, default=0.01)
    args = parser.parse_args()

    direction = {}
    for mode in MODES:
        direction[mode] = json.loads(
            (args.direction_dir / f"{mode}_pan_direction_consistency.json").read_text(encoding="utf-8")
        )
    gt_slopes = {mode: direction[mode]["observational"]["gt"]["slope"] for mode in MODES}
    if len({round(value, 12) for value in gt_slopes.values()}) != 1:
        raise ValueError(f"GT slopes disagree across modes: {gt_slopes}")
    gt_slope = next(iter(gt_slopes.values()))
    if gt_slope is None or gt_slope == 0:
        raise ValueError(f"GT slope is not directional: {gt_slope}")
    increasing = gt_slope > 0

    result = {
        "status": "failed",
        "direction_reference": {
            "source": "full-trajectory independent pan-direction audit",
            "gt_slope": gt_slope,
            "expected_cx_to_pan": "increasing" if increasing else "decreasing",
        },
        "modes": {},
    }
    for mode in MODES:
        rows = []
        for step in STEPS:
            row = json.loads((args.eval_dir / f"{mode}_{step}.json").read_text(encoding="utf-8"))
            values = row["coordinate_only_h0_pan"]["normal"]
            direction_pass = strictly_monotonic(values, increasing)
            gates = row["gates"]
            span_pass = gates["coordinate_only_span"] >= args.min_pan_span
            effect_pass = max(gates["zero_h0_pan_effect"], gates["reverse_h0_pan_effect"]) >= args.min_box_effect
            residual = row.get("residual")
            residual_pass = mode != "C2" or (
                residual is not None and 0.01 <= residual["alpha"] <= 0.95
            )
            rows.append(
                {
                    "step": step,
                    "direction_pass": direction_pass,
                    "span": gates["coordinate_only_span"],
                    "span_pass": span_pass,
                    "box_effect_pass": effect_pass,
                    "residual_pass": residual_pass,
                    "coordinate_only_h0_pan": values,
                }
            )
        final = rows[-1]
        passed = all(row["direction_pass"] for row in rows) and final["span_pass"] and final["box_effect_pass"] and final["residual_pass"]
        result["modes"][mode] = {"passed": passed, "checkpoints": rows}

    result["status"] = "passed" if any(item["passed"] for item in result["modes"].values()) else "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
