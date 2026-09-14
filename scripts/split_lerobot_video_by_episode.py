#!/usr/bin/env python
"""Split a LeRobot v3 video into frame-exact episode clips for annotation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="LeRobot v3 dataset directory")
    parser.add_argument(
        "--video-key",
        default="observation.images.top",
        help="Video feature to split (default: observation.images.top)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: <dataset>/annotation_segments/<video-key>)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=0,
        help="H.264 quality; 0 is lossless (default: 0)",
    )
    parser.add_argument(
        "--copy-codec",
        action="store_true",
        help="Copy the source video stream without re-encoding; episode boundaries must be keyframes",
    )
    return parser.parse_args()


def load_episodes(dataset: Path, video_key: str) -> list[dict[str, int | float]]:
    metadata_path = dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    timestamp_prefix = f"videos/{video_key}"
    columns = [
        "episode_index",
        "length",
        "dataset_from_index",
        "dataset_to_index",
        f"{timestamp_prefix}/from_timestamp",
        f"{timestamp_prefix}/to_timestamp",
    ]
    table = pq.read_table(metadata_path, columns=columns).to_pydict()
    episodes = [{column: table[column][row] for column in columns} for row in range(len(table["episode_index"]))]

    for previous, current in zip(episodes, episodes[1:], strict=False):
        if previous["dataset_to_index"] != current["dataset_from_index"]:
            raise ValueError("Episode dataset indices are not contiguous")
    if any(ep["dataset_to_index"] - ep["dataset_from_index"] != ep["length"] for ep in episodes):
        raise ValueError("Episode length disagrees with dataset frame indices")
    return episodes


def find_video(dataset: Path, video_key: str) -> Path:
    metadata_path = dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(
        metadata_path,
        columns=[f"videos/{video_key}/chunk_index", f"videos/{video_key}/file_index"],
    ).to_pydict()
    locations = set(zip(*table.values(), strict=False))
    if len(locations) != 1:
        raise ValueError("This utility expects all requested episodes to share one source video")
    chunk_index, file_index = locations.pop()
    return dataset / "videos" / video_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"


def probe_frame_count(video_path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    episodes = load_episodes(dataset, args.video_key)
    source_path = find_video(dataset, args.video_key)
    output_dir = (args.output_dir or dataset / "annotation_segments" / args.video_key).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    boundaries = [int(ep["dataset_to_index"]) for ep in episodes[:-1]]
    with (dataset / "meta" / "info.json").open(encoding="utf-8") as info_file:
        fps = float(json.load(info_file)["fps"])
    force_keyframe_times = ",".join(f"{boundary / fps:.9f}" for boundary in boundaries)
    segment_frames = ",".join(str(boundary) for boundary in boundaries)
    codec_args = ["-c:v", "copy"] if args.copy_codec else [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-force_key_frames",
        force_keyframe_times,
    ]
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "warning",
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-an",
            *codec_args,
            "-f",
            "segment",
            "-segment_frames",
            segment_frames,
            "-reset_timestamps",
            "1",
            str(output_dir / "episode_%03d.mp4"),
        ],
        check=True,
    )
    clip_paths = sorted(output_dir.glob("episode_*.mp4"))
    if len(clip_paths) != len(episodes):
        raise ValueError(f"Expected {len(episodes)} clips, found {len(clip_paths)}")
    written_counts = [probe_frame_count(path) for path in clip_paths]
    expected_counts = [int(ep["length"]) for ep in episodes]
    if written_counts != expected_counts:
        raise ValueError(f"Frame counts differ from metadata: expected {expected_counts}, got {written_counts}")

    prefix = f"videos/{args.video_key}"
    with (output_dir / "segments.csv").open("w", newline="", encoding="utf-8-sig") as manifest:
        writer = csv.DictWriter(
            manifest,
            fieldnames=[
                "episode_index",
                "file",
                "frame_count",
                "dataset_from_index",
                "dataset_to_index_exclusive",
                "source_from_timestamp",
                "source_to_timestamp",
            ],
        )
        writer.writeheader()
        for episode, frame_count in zip(episodes, written_counts, strict=True):
            writer.writerow(
                {
                    "episode_index": episode["episode_index"],
                    "file": f"episode_{int(episode['episode_index']):03d}.mp4",
                    "frame_count": frame_count,
                    "dataset_from_index": episode["dataset_from_index"],
                    "dataset_to_index_exclusive": episode["dataset_to_index"],
                    "source_from_timestamp": episode[f"{prefix}/from_timestamp"],
                    "source_to_timestamp": episode[f"{prefix}/to_timestamp"],
                }
            )

    print(f"Wrote {len(episodes)} clips ({sum(written_counts)} frames) to {output_dir}")


if __name__ == "__main__":
    main()
