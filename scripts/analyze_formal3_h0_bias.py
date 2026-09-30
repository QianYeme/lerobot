"""Train48 frame0 h0 bias analysis for the h0-action-aux screening.

Reads one checkpoint's train48 frame0 eval npz directory and reports, per
episode and aggregated:
- the same reset-observation metrics as the preroll P3 gates (cosine+/pan/
  norm ratio, first 5 action dims, baseline = median of original actions
  0..14, desired = original action at trim start S);
- the raw per-joint pred-target bias at h0 in degrees (the systematic fixed
  bias diagnosed on the preroll 10k), plus mean |bias|;
- which truth chunk row the predicted h0 is closest to (row 0 expected).

This is diagnostic only: it is not a promotion gate and never touches dev12.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

ARM = 5


def load_npz(directory: Path, episode: int) -> dict[str, np.ndarray]:
    with np.load(directory / f"episode_{episode:03d}_chunks.npz") as data:
        return {key: data[key] for key in data.files}


def load_original_actions(original_root: Path, episodes: list[int]) -> dict[int, np.ndarray]:
    table = pq.read_table(original_root / "data/chunk-000/file-000.parquet",
                          columns=["episode_index", "frame_index", "action"])
    actions = {}
    for ep in episodes:
        rows = table.filter(pc.equal(table["episode_index"], ep)).sort_by("frame_index")
        actions[ep] = np.stack(rows["action"].to_numpy())
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-dir", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--timetrim-root", type=Path, required=True)
    parser.add_argument("--episodes", type=str, required=True,
                        help="comma-separated episode indices present in the npz dir")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    episodes = [int(x) for x in args.episodes.split(",") if x.strip()]
    boundaries = {row["episode_index"]: row
                  for row in json.loads((args.timetrim_root / "meta/time_trim_manifest.json").read_text())["boundaries"]}
    orig_actions = load_original_actions(args.original_root, episodes)

    per_episode = {}
    truth_sanity_failures = []
    for ep in episodes:
        data = load_npz(args.npz_dir, ep)
        pred_h0 = data["prediction_raw"][0, 0]
        truth_chunk = data["truth_raw"][0]
        start = boundaries[ep]["start"]
        desired = orig_actions[ep][start]
        if not np.allclose(truth_chunk[0], desired):
            truth_sanity_failures.append(ep)
        baseline = np.median(orig_actions[ep][0:15], axis=0)
        pred_delta = pred_h0[:ARM] - baseline[:ARM]
        desired_delta = desired[:ARM] - baseline[:ARM]
        desired_norm = float(np.linalg.norm(desired_delta))
        pred_norm = float(np.linalg.norm(pred_delta))
        denominator = desired_norm * pred_norm
        closest = int(np.argmin(np.linalg.norm(truth_chunk[:30, :ARM] - pred_h0[:ARM], axis=1)))
        # The formal3 action columns store joint angles in degrees natively;
        # report the raw pred-target bias directly (already in degrees).
        per_episode[str(ep)] = {
            "trim_start_S": start,
            "pan_direction_ok": bool(np.sign(pred_delta[0]) == np.sign(desired_delta[0]) and np.sign(desired_delta[0]) != 0),
            "motion_cosine": float(np.dot(desired_delta, pred_delta) / denominator) if denominator else np.nan,
            "motion_norm_ratio": float(pred_norm / desired_norm) if desired_norm else np.nan,
            "bias_deg_arm": (pred_h0[:ARM] - truth_chunk[0, :ARM]).tolist(),
            "bias_deg_gripper": float(pred_h0[5] - truth_chunk[0, 5]),
            "closest_truth_row_in_first30": closest,
        }

    values = list(per_episode.values())
    cosines = np.asarray([v["motion_cosine"] for v in values])
    pans = np.asarray([v["pan_direction_ok"] for v in values])
    ratios = np.asarray([v["motion_norm_ratio"] for v in values])
    bias_arm = np.asarray([v["bias_deg_arm"] for v in values])
    closest_rows = np.asarray([v["closest_truth_row_in_first30"] for v in values])

    report = {
        "npz_dir": str(args.npz_dir),
        "episodes": episodes,
        "truth_sanity_failures": truth_sanity_failures,
        "metrics": {
            "cosine_positive_count": int(np.sum(cosines > 0)),
            "cosine_mean": float(np.mean(cosines)),
            "pan_direction_count": int(np.sum(pans)),
            "norm_ratio_median": float(np.median(ratios)),
            "bias_deg_mean": np.mean(bias_arm, axis=0).tolist(),
            "bias_deg_abs_mean_per_joint": np.mean(np.abs(bias_arm), axis=0).tolist(),
            "bias_deg_abs_mean_arm": float(np.mean(np.abs(bias_arm))),
            "bias_deg_gripper_mean": float(np.mean([v["bias_deg_gripper"] for v in values])),
            "closest_row_histogram": {str(row): int(np.sum(closest_rows == row)) for row in sorted(set(closest_rows.tolist()))},
        },
        "per_episode": per_episode,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = args.out.with_suffix(".md")
    m = report["metrics"]
    lines = [f"# h0 bias analysis: {Path(args.npz_dir).name}", "",
             "| metric | value |", "|---|---|",
             f"| episodes | {len(episodes)} |",
             f"| truth row 0 == action[S] failures | {truth_sanity_failures} |",
             f"| cosine+ | {m['cosine_positive_count']}/{len(episodes)} |",
             f"| pan direction | {m['pan_direction_count']}/{len(episodes)} |",
             f"| norm ratio median | {m['norm_ratio_median']:.3f} |",
             f"| bias deg mean (arm joints) | {[round(x, 2) for x in m['bias_deg_mean']]} |",
             f"| bias deg abs mean (arm joints) | {[round(x, 2) for x in m['bias_deg_abs_mean_per_joint']]} |",
             f"| bias deg abs mean arm (scalar) | {m['bias_deg_abs_mean_arm']:.2f} |",
             f"| bias deg gripper mean | {m['bias_deg_gripper_mean']:.2f} |",
             f"| closest truth row histogram (first 30) | {m['closest_row_histogram']} |"]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
