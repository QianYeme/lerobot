"""Validate decoded video alignment and row metadata for Formal3 reset-preroll."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from lerobot.datasets.video_utils import decode_video_frames


def read_frame(path: Path, index: int, fps: int) -> np.ndarray:
    frame = decode_video_frames(path, [index / fps], tolerance_s=1e-4, backend="pyav")[0]
    return frame.numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--derived", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 10, 37, 59])
    args = parser.parse_args()

    original_info = json.loads((args.original / "meta/info.json").read_text(encoding="utf-8"))
    derived_info = json.loads((args.derived / "meta/info.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.derived / "meta/reset_preroll_manifest.json").read_text(encoding="utf-8"))
    mapping = {row["episode_index"]: row for row in manifest["episodes"]}
    source_episodes = pq.read_table(args.original / "meta/episodes/chunk-000/file-000.parquet")
    derived_data = pq.read_table(args.derived / "data/chunk-000/file-000.parquet")
    phase = pq.read_table(args.derived / "meta/reviewed_phase_labels.parquet")
    fps = int(derived_info["fps"])

    comparisons = []
    for episode in args.episodes:
        source_meta = source_episodes.filter(pc.equal(source_episodes["episode_index"], episode)).to_pylist()[0]
        row = mapping[episode]
        prefix, start, end = row["prefix_frames"], row["start"], row["end"]
        pairs = [(0, 0), (prefix - 1, prefix - 1), (prefix, start)]
        if start + 14 < end:
            pairs.append((prefix + 14, start + 14))
        pairs.append((row["length"] - 1, end - 1))
        for camera in ("observation.images.top", "observation.images.gripper"):
            key = f"videos/{camera}"
            source_path = args.original / original_info["video_path"].format(
                video_key=camera,
                chunk_index=source_meta[f"{key}/chunk_index"],
                file_index=source_meta[f"{key}/file_index"],
            )
            source_offset = round(float(source_meta[f"{key}/from_timestamp"]) * fps)
            derived_path = args.derived / derived_info["video_path"].format(
                video_key=camera, chunk_index=0, file_index=episode,
            )
            for derived_frame, source_frame in pairs:
                actual = read_frame(derived_path, derived_frame, fps).astype(np.float32)
                expected = read_frame(source_path, source_offset + source_frame, fps).astype(np.float32)
                diff = np.abs(actual - expected)
                comparisons.append({
                    "episode": episode, "camera": camera,
                    "derived_frame": derived_frame, "source_frame": source_frame,
                    "mean_abs_diff_255": float(diff.mean() * 255),
                    "max_abs_diff_255": float(diff.max() * 255),
                })

    data_checks = {
        "total_frames": derived_data.num_rows == derived_info["total_frames"] == manifest["frames"],
        "phase_rows": phase.num_rows == derived_data.num_rows,
        "frame0_per_episode": all(
            derived_data.filter(pc.equal(derived_data["episode_index"], episode))["frame_index"][0].as_py() == 0
            for episode in range(60)
        ),
        "decoded_mean_abs_diff_le_5": max(row["mean_abs_diff_255"] for row in comparisons) <= 5.0,
    }
    result = {
        "status": "PASS" if all(data_checks.values()) else "FAIL",
        "checks": data_checks,
        "max_mean_abs_diff_255": max(row["mean_abs_diff_255"] for row in comparisons),
        "max_abs_diff_255": max(row["max_abs_diff_255"] for row in comparisons),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "checks", "max_mean_abs_diff_255", "max_abs_diff_255")}, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
