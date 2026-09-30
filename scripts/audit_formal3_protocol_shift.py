"""Audit Formal3 acquisition-protocol differences relevant to reset-pair failures."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


ARM = 5


def first_box_center(path: Path) -> tuple[float, float]:
    box = ET.parse(path).find('.//box[@frame="0"]')
    if box is None:
        raise ValueError(f"Missing frame-0 box: {path}")
    return (
        (float(box.attrib["xtl"]) + float(box.attrib["xbr"])) / 2,
        (float(box.attrib["ytl"]) + float(box.attrib["ybr"])) / 2,
    )


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def standardized_difference(left: np.ndarray, right: np.ndarray) -> float:
    pooled = np.sqrt((np.var(left) + np.var(right)) / 2)
    return float((np.mean(left) - np.mean(right)) / pooled) if pooled else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--trim-manifest", type=Path, required=True)
    parser.add_argument("--overshoot-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    table = pq.read_table(args.dataset / "data/chunk-000/file-000.parquet")
    boundaries = {
        row["episode_index"]: row
        for row in json.loads(args.trim_manifest.read_text(encoding="utf-8"))["boundaries"]
    }
    diagnosis = json.loads(args.overshoot_json.read_text(encoding="utf-8"))["per_episode"]

    rows = []
    for episode in range(60):
        episode_table = table.filter(pc.equal(table["episode_index"], episode))
        actions = np.asarray(episode_table["action"].to_pylist())
        states = np.asarray(episode_table["observation.state"].to_pylist())
        start = boundaries[episode]["start"]
        baseline = np.median(actions[:15], axis=0)
        first_delta = actions[start, :ARM] - baseline[:ARM]
        ramp = np.linalg.norm(np.diff(actions[start : start + 15, :ARM], axis=0), axis=1)
        cup_cx, cup_cy = first_box_center(
            args.dataset / f"annotations/top/episode_{episode:03d}.xml"
        )
        row = {
            "episode": episode,
            "protocol_group": "ep0-37" if episode <= 37 else "ep38-59",
            "split": "development" if str(episode) in diagnosis else "train",
            "trim_start_S": start,
            "cup_cx": cup_cx,
            "cup_cy": cup_cy,
            "reset_state_arm": states[0, :ARM].tolist(),
            "first_motion_norm": float(np.linalg.norm(first_delta)),
            "first_motion_pan": float(first_delta[0]),
            "ramp_speed_mean": float(np.mean(ramp)),
            "ramp_speed_max": float(np.max(ramp)),
        }
        if str(episode) in diagnosis:
            row["resetpair_ratio"] = diagnosis[str(episode)]["vector_ratio"]
            row["resetpair_cosine"] = diagnosis[str(episode)]["cosine"]
        rows.append(row)

    scalar_fields = [
        "trim_start_S", "cup_cx", "cup_cy", "first_motion_norm",
        "first_motion_pan", "ramp_speed_mean", "ramp_speed_max",
    ]
    groups = {}
    for name in ("ep0-37", "ep38-59"):
        selected = [row for row in rows if row["protocol_group"] == name]
        groups[name] = {field: summarize(np.asarray([row[field] for row in selected])) for field in scalar_fields}
        arm = np.asarray([row["reset_state_arm"] for row in selected])
        groups[name]["reset_state_arm"] = [summarize(arm[:, joint]) for joint in range(ARM)]

    left = [row for row in rows if row["protocol_group"] == "ep0-37"]
    right = [row for row in rows if row["protocol_group"] == "ep38-59"]
    effects = {}
    for field in scalar_fields:
        effects[field] = standardized_difference(
            np.asarray([row[field] for row in left]), np.asarray([row[field] for row in right])
        )
    left_state = np.asarray([row["reset_state_arm"] for row in left])
    right_state = np.asarray([row["reset_state_arm"] for row in right])
    effects["reset_state_arm"] = [
        standardized_difference(left_state[:, joint], right_state[:, joint]) for joint in range(ARM)
    ]

    left_s = np.asarray([row["trim_start_S"] for row in left])
    right_s = np.asarray([row["trim_start_S"] for row in right])
    overlap = [float(max(left_s.min(), right_s.min())), float(min(left_s.max(), right_s.max()))]

    dev = [row for row in rows if row["split"] == "development"]
    design = np.asarray([
        [1.0, row["trim_start_S"], float(row["protocol_group"] == "ep38-59")]
        for row in dev
    ])
    target = np.asarray([row["resetpair_ratio"] for row in dev])
    coefficients, _, rank, singular = np.linalg.lstsq(design, target, rcond=None)

    report = {
        "schema_version": 1,
        "episodes": rows,
        "group_summaries": groups,
        "standardized_mean_differences_ep0_37_minus_ep38_59": effects,
        "trim_start_overlap": overlap,
        "dev12_ratio_regression": {
            "formula": "resetpair_ratio = intercept + beta_S*S + beta_group*I(ep38-59)",
            "coefficients": {
                "intercept": float(coefficients[0]),
                "beta_S": float(coefficients[1]),
                "beta_group": float(coefficients[2]),
            },
            "rank": int(rank),
            "singular_values": singular.tolist(),
            "warning": "Exploratory only: n=12, group/cup/reset-state covariates remain entangled.",
        },
        "conclusion": (
            "The full 60-episode S ranges overlap, while reset pose and cup-position distributions "
            "show large protocol shifts. Dev12 alone cannot identify S as the unique overshoot cause."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "protocol_shift_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Formal3 protocol-shift audit",
        "",
        "This is a read-only data audit; it does not establish a causal effect on robot success.",
        "",
        "| metric | ep0-37 mean | ep38-59 mean | standardized difference |",
        "|---|---:|---:|---:|",
    ]
    for field in scalar_fields:
        lines.append(
            f"| {field} | {groups['ep0-37'][field]['mean']:.3f} | "
            f"{groups['ep38-59'][field]['mean']:.3f} | {effects[field]:.3f} |"
        )
    for joint in range(ARM):
        lines.append(
            f"| reset_state_arm[{joint}] | {groups['ep0-37']['reset_state_arm'][joint]['mean']:.3f} | "
            f"{groups['ep38-59']['reset_state_arm'][joint]['mean']:.3f} | "
            f"{effects['reset_state_arm'][joint]:.3f} |"
        )
    lines += [
        "",
        f"- trim-start S overlap: {overlap[0]:.0f}..{overlap[1]:.0f}",
        f"- exploratory dev12 beta_S: {coefficients[1]:.4f}",
        f"- exploratory dev12 beta_group: {coefficients[2]:.4f}",
        "- conclusion: full-dataset S is not separated by protocol; reset pose and cup position shift strongly.",
        "- limitation: the acquisition change jointly affects pose, cup distribution, and possibly camera appearance, so this audit does not identify one unique causal variable.",
    ]
    (args.output / "protocol_shift_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output / "protocol_shift_audit.json")


if __name__ == "__main__":
    main()
