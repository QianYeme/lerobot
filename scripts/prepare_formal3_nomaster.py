"""Create an immutable 8-state formal3 derivative without master_gripper.pos."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


STATE_KEY = "observation.state"
REMOVED_FEATURE = "master_gripper.pos"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def vector_stats(values: np.ndarray) -> dict[str, list]:
    values = np.asarray(values, dtype=np.float64)
    result = {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "count": np.array([len(values)]),
    }
    result.update(
        {f"q{int(quantile * 100):02d}": np.quantile(values, quantile, axis=0)
         for quantile in (0.01, 0.1, 0.5, 0.9, 0.99)}
    )
    return {key: value.tolist() for key, value in result.items()}


def prepare(source: Path, destination: Path, phase_labels: Path) -> dict:
    source = source.resolve()
    destination = destination.resolve()
    phase_labels = phase_labels.resolve()
    staging = destination.with_name(f"{destination.name}.preparing")
    if destination.exists() or staging.exists():
        raise FileExistsError(f"Refusing to overwrite {destination} or {staging}")

    info_path = source / "meta/info.json"
    stats_path = source / "meta/stats.json"
    data_paths = sorted((source / "data").glob("*/*.parquet"))
    episode_paths = sorted((source / "meta/episodes").glob("*/*.parquet"))
    if not data_paths or not episode_paths or not phase_labels.is_file():
        raise FileNotFoundError("Source data, episode metadata, or frozen phase labels are missing")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    state_names = info["features"][STATE_KEY]["names"]
    if state_names != [
        "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos",
        "wrist_roll.pos", "gripper.pos", "gripper.load", "gripper.curr", REMOVED_FEATURE,
    ]:
        raise ValueError(f"Unexpected source state schema: {state_names}")

    data = pa.concat_tables([pq.read_table(path) for path in data_paths])
    episodes = pa.concat_tables([pq.read_table(path) for path in episode_paths])
    states_9d = np.asarray(data[STATE_KEY].to_pylist(), dtype=np.float32)
    states_8d = states_9d[:, :8].copy()
    actions = np.asarray(data["action"].to_pylist(), dtype=np.float32)
    if states_9d.shape != (info["total_frames"], 9) or actions.shape != (info["total_frames"], 6):
        raise ValueError("Source state/action dimensions disagree with metadata")
    if not np.isfinite(states_8d).all() or not np.isfinite(actions).all():
        raise ValueError("Source contains non-finite state or action values")
    if len(episodes) != info["total_episodes"]:
        raise ValueError("Episode metadata count disagrees with info.json")
    if data["index"].to_pylist() != list(range(info["total_frames"])):
        raise ValueError("Global frame indices are not contiguous")

    expected_keys = set(zip(data["episode_index"].to_pylist(), data["frame_index"].to_pylist(), strict=True))
    label_table = pq.read_table(
        phase_labels, columns=["episode_index", "frame_index", "phase_id", "valid", "reviewed"]
    )
    label_keys = set(zip(label_table["episode_index"].to_pylist(), label_table["frame_index"].to_pylist(), strict=True))
    if len(label_keys) != len(label_table) or label_keys != expected_keys:
        raise ValueError("Frozen phase labels do not map one-to-one onto source frames")
    if not all(label_table["reviewed"].to_pylist()):
        raise ValueError("Frozen phase labels contain unreviewed rows")

    source_hashes = {
        str(path.relative_to(source)): sha256(path)
        for path in (info_path, stats_path, source / "meta/tasks.parquet", *data_paths, *episode_paths)
    }
    data_8d = data.set_column(
        data.schema.get_field_index(STATE_KEY),
        STATE_KEY,
        pa.array(states_8d.tolist(), type=pa.list_(pa.float32())),
    )
    for column in episodes.column_names:
        if column.startswith(f"stats/{STATE_KEY}/") and not column.endswith("/count"):
            values = [value[:8] for value in episodes[column].to_pylist()]
            episodes = episodes.set_column(
                episodes.schema.get_field_index(column),
                column,
                pa.array(values, type=episodes.schema.field(column).type),
            )

    info["features"][STATE_KEY]["names"] = state_names[:8]
    info["features"][STATE_KEY]["shape"] = [8]
    stats[STATE_KEY] = vector_stats(states_8d)

    (staging / "data/chunk-000").mkdir(parents=True)
    (staging / "meta/episodes/chunk-000").mkdir(parents=True)
    pq.write_table(data_8d.replace_schema_metadata(None), staging / "data/chunk-000/file-000.parquet")
    pq.write_table(episodes.replace_schema_metadata(None), staging / "meta/episodes/chunk-000/file-000.parquet")
    (staging / "meta/info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (staging / "meta/stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tasks_source = source / "meta/tasks.parquet"
    tasks_destination = staging / "meta/tasks.parquet"
    os.link(tasks_source, tasks_destination)

    linked_videos: dict[str, str] = {}
    cameras = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    for camera in cameras:
        pairs = set(zip(
            episodes[f"videos/{camera}/chunk_index"].to_pylist(),
            episodes[f"videos/{camera}/file_index"].to_pylist(),
            strict=True,
        ))
        for chunk_index, file_index in sorted(pairs):
            relative = Path(info["video_path"].format(
                video_key=camera, chunk_index=chunk_index, file_index=file_index
            ))
            video_source = source / relative
            video_destination = staging / relative
            video_destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(video_source, video_destination)
            if not os.path.samefile(video_source, video_destination):
                raise AssertionError(f"Video is not hard-linked: {relative}")
            linked_videos[str(relative)] = sha256(video_source)

    written = pq.read_table(staging / "data/chunk-000/file-000.parquet")
    written_states = np.asarray(written[STATE_KEY].to_pylist(), dtype=np.float32)
    if not np.array_equal(written_states, states_8d):
        raise AssertionError("Written 8D states differ from source first eight dimensions")
    for column in data.column_names:
        if column != STATE_KEY and not written[column].equals(data[column]):
            raise AssertionError(f"Unexpected changed data column: {column}")
    if any(sha256(source / relative) != digest for relative, digest in source_hashes.items()):
        raise AssertionError("Source dataset changed during preparation")

    manifest = {
        "schema_version": 1,
        "source": str(source),
        "destination": str(destination),
        "removed_feature": REMOVED_FEATURE,
        "state_names": state_names[:8],
        "action_names": info["features"]["action"]["names"],
        "episodes": info["total_episodes"],
        "frames": info["total_frames"],
        "statistics_scope": "all 60 episodes, matching the source train split; P3 must freeze a train-only split/statistics variant",
        "phase_labels": {"path": str(phase_labels), "sha256": sha256(phase_labels)},
        "source_hashes": source_hashes,
        "video_sha256": linked_videos,
        "videos": "hard-linked without re-encoding",
        "checks": {
            "state_9_to_8": True,
            "non_state_data_columns_exact": True,
            "phase_keys_one_to_one": True,
            "all_phase_rows_reviewed": True,
            "source_unchanged": True,
        },
    }
    manifest_path = staging / "meta/nomaster_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--phase-labels", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination, args.phase_labels), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
