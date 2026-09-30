"""Create a non-destructive Formal3 derivative with idle time removed.

The source videos are hard-linked without re-encoding.  Episode video offsets
are shifted so that derived frame zero still decodes the original start frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


NUMERIC_KEYS = ("action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index")
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def feature_stats(values: np.ndarray) -> dict[str, list]:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    result = {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "count": np.asarray([len(values)], dtype=np.int64),
    }
    result.update({f"q{int(q * 100):02d}": np.quantile(values, q, axis=0) for q in QUANTILES})
    return {key: value.tolist() for key, value in result.items()}


def find_motion_start(actions: np.ndarray, threshold: float, sustain: int, baseline_frames: int) -> int:
    arm = np.asarray(actions, dtype=np.float64)[:, :5]
    baseline = np.median(arm[:baseline_frames], axis=0)
    moving = np.max(np.abs(arm - baseline), axis=1) >= threshold
    for start in range(len(moving) - sustain + 1):
        if bool(np.all(moving[start : start + sustain])):
            return start
    raise ValueError("No sustained arm departure found")


def build_boundaries(
    data: pa.Table, reviews: dict, *, threshold: float, sustain: int, baseline_frames: int
) -> list[dict[str, int]]:
    result = []
    episodes = reviews.get("episodes", {})
    for episode_index in sorted(set(data["episode_index"].to_pylist())):
        mask = np.asarray(data["episode_index"].to_numpy()) == episode_index
        actions = np.asarray(data["action"].to_pylist(), dtype=np.float32)[mask]
        review = episodes.get(str(episode_index))
        if not review or review.get("reviewed") is not True:
            raise ValueError(f"Episode {episode_index} has no completed human review")
        start = find_motion_start(actions, threshold, sustain, baseline_frames)
        end = int(review["steps"]["release"]["end_frame"]) + 1
        if not 0 <= start < end <= len(actions):
            raise ValueError(f"Invalid trim interval for episode {episode_index}: [{start}, {end})")
        result.append({"episode_index": episode_index, "start": start, "end": end, "length": end - start})
    return result


def trim_xml(source: Path, destination: Path, start: int, end: int) -> int:
    tree = ET.parse(source)
    root = tree.getroot()
    kept = 0
    for track in root.findall(".//track"):
        for box in list(track.findall("box")):
            frame = int(box.attrib["frame"])
            if start <= frame < end:
                box.attrib["frame"] = str(frame - start)
                kept += 1
            else:
                track.remove(box)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return kept


def replace_column(table: pa.Table, name: str, values) -> pa.Table:
    index = table.schema.get_field_index(name)
    return table.set_column(index, name, pa.array(values, type=table.schema.field(index).type))


def prepare(args: argparse.Namespace) -> dict:
    source = args.source.resolve()
    destination = args.destination.resolve()
    staging = destination.with_name(destination.name + ".preparing")
    if destination.exists() or staging.exists():
        raise FileExistsError(f"Refusing to overwrite {destination} or {staging}")

    data_paths = sorted((source / "data").glob("*/*.parquet"))
    episode_paths = sorted((source / "meta/episodes").glob("*/*.parquet"))
    if not data_paths or not episode_paths:
        raise FileNotFoundError("Source data or episode metadata is missing")
    data = pa.concat_tables([pq.read_table(path) for path in data_paths])
    episodes = pa.concat_tables([pq.read_table(path) for path in episode_paths])
    info = json.loads((source / "meta/info.json").read_text(encoding="utf-8"))
    reviews = json.loads(args.reviews.read_text(encoding="utf-8"))
    boundaries = build_boundaries(
        data, reviews, threshold=args.motion_threshold, sustain=args.sustain_frames,
        baseline_frames=args.baseline_frames,
    )
    by_episode = {row["episode_index"]: row for row in boundaries}

    phase = pq.read_table(args.phase_labels)
    source_phase_keys = set(zip(phase["episode_index"].to_pylist(), phase["frame_index"].to_pylist(), strict=True))
    source_keys = set(zip(data["episode_index"].to_pylist(), data["frame_index"].to_pylist(), strict=True))
    if source_phase_keys != source_keys:
        raise ValueError("Phase labels do not map one-to-one onto source frames")

    trimmed_parts: list[pa.Table] = []
    phase_parts: list[pa.Table] = []
    new_episode_rows = []
    global_index = 0
    fps = int(info["fps"])
    camera_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    for boundary in boundaries:
        ep = boundary["episode_index"]
        start, end, length = boundary["start"], boundary["end"], boundary["length"]
        ep_data = data.filter(pa.compute.equal(data["episode_index"], ep)).slice(start, length)
        ep_data = replace_column(ep_data, "timestamp", np.arange(length, dtype=np.float32) / fps)
        ep_data = replace_column(ep_data, "frame_index", np.arange(length, dtype=np.int64))
        ep_data = replace_column(ep_data, "index", np.arange(global_index, global_index + length, dtype=np.int64))
        trimmed_parts.append(ep_data)

        ep_phase = phase.filter(pa.compute.equal(phase["episode_index"], ep)).slice(start, length)
        ep_phase = replace_column(ep_phase, "frame_index", np.arange(length, dtype=np.int64))
        phase_parts.append(ep_phase)

        old = episodes.filter(pa.compute.equal(episodes["episode_index"], ep)).to_pylist()[0]
        old["length"] = length
        old["dataset_from_index"] = global_index
        old["dataset_to_index"] = global_index + length
        for camera in camera_keys:
            prefix = f"videos/{camera}"
            shifted = float(old[f"{prefix}/from_timestamp"]) + start / fps
            old[f"{prefix}/from_timestamp"] = shifted
            old[f"{prefix}/to_timestamp"] = shifted + length / fps
        for key in NUMERIC_KEYS:
            stats = feature_stats(np.asarray(ep_data[key].to_pylist()))
            for stat, value in stats.items():
                column = f"stats/{key}/{stat}"
                if column in old:
                    old[column] = value
        new_episode_rows.append(old)
        global_index += length

    trimmed = pa.concat_tables(trimmed_parts)
    trimmed_phase = pa.concat_tables(phase_parts)
    new_episodes = pa.Table.from_pylist(new_episode_rows, schema=episodes.schema)
    info["total_frames"] = len(trimmed)

    (staging / "data/chunk-000").mkdir(parents=True)
    (staging / "meta/episodes/chunk-000").mkdir(parents=True)
    pq.write_table(trimmed.replace_schema_metadata(None), staging / "data/chunk-000/file-000.parquet")
    pq.write_table(new_episodes.replace_schema_metadata(None), staging / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(trimmed_phase.replace_schema_metadata(None), staging / "meta/reviewed_phase_labels.parquet")
    (staging / "meta/info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(source / "meta/tasks.parquet", staging / "meta/tasks.parquet")

    global_stats = json.loads((source / "meta/stats.json").read_text(encoding="utf-8"))
    for key in NUMERIC_KEYS:
        global_stats[key] = feature_stats(np.asarray(trimmed[key].to_pylist()))
    (staging / "meta/stats.json").write_text(json.dumps(global_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    linked_videos = 0
    for video in (source / "videos").rglob("*.mp4"):
        target = staging / video.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(video, target)
        linked_videos += 1

    annotation_boxes = 0
    for boundary in boundaries:
        ep = boundary["episode_index"]
        xml = source / f"annotations/top/episode_{ep:03d}.xml"
        annotation_boxes += trim_xml(
            xml, staging / xml.relative_to(source), boundary["start"], boundary["end"]
        )
    if (source / "annotations/manifest.json").is_file():
        shutil.copy2(source / "annotations/manifest.json", staging / "annotations/manifest.source.json")

    checks = {
        "episodes_preserved": len(boundaries) == info["total_episodes"],
        "rows_match_boundaries": len(trimmed) == sum(item["length"] for item in boundaries),
        "global_index_contiguous": trimmed["index"].to_pylist() == list(range(len(trimmed))),
        "phase_rows_match": len(trimmed_phase) == len(trimmed),
        "video_files_hardlinked": linked_videos > 0,
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    manifest = {
        "schema_version": 1,
        "source": str(source),
        "source_info_sha256": sha256(source / "meta/info.json"),
        "reviews": {"path": str(args.reviews.resolve()), "sha256": sha256(args.reviews)},
        "phase_labels": {"path": str(args.phase_labels.resolve()), "sha256": sha256(args.phase_labels)},
        "motion_start_rule": {"arm_linf_degrees": args.motion_threshold, "sustain_frames": args.sustain_frames, "baseline_frames": args.baseline_frames},
        "end_rule": "one frame after human-reviewed release.end_frame",
        "source_frames": len(data), "retained_frames": len(trimmed),
        "retained_fraction": len(trimmed) / len(data), "boundaries": boundaries,
        "videos": f"{linked_videos} hard-linked files; episode from_timestamp shifted; no re-encoding",
        "annotation_boxes_retained": annotation_boxes,
        "camera_statistics": "copied from source; numerical feature statistics recomputed on trimmed rows",
        "checks": checks,
    }
    (staging / "meta/time_trim_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--phase-labels", type=Path, required=True)
    parser.add_argument("--motion-threshold", type=float, default=5.0)
    parser.add_argument("--sustain-frames", type=int, default=3)
    parser.add_argument("--baseline-frames", type=int, default=15)
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
