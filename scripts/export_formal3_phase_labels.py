"""Export auditable four-phase candidate labels from formal3 gripper telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PHASE_IDS = {"unknown": -1, "approach": 0, "grasp": 1, "lift_place": 2, "release": 3}


def first_sustained(values: np.ndarray, start: int, threshold: float, *, above: bool, frames: int) -> int | None:
    predicate = np.greater_equal if above else np.less_equal
    for index in range(max(0, start), len(values) - frames + 1):
        if bool(np.all(predicate(values[index : index + frames], threshold))):
            return index
    return None


def motion_interval(
    values: np.ndarray,
    start: int,
    pivot: int,
    stop: int,
    *,
    opening: bool,
    sustain_frames: int,
    margin: float = 0.1,
) -> tuple[int | None, int | None]:
    """Estimate 10%-to-90% gripper travel within one commanded motion."""
    if not 0 <= start < pivot < stop <= len(values):
        return None, None
    before = values[start:pivot]
    after = values[pivot:stop]
    high = float(np.percentile(before if not opening else after, 90))
    low = float(np.percentile(after if not opening else before, 10))
    travel = high - low
    if travel <= 0:
        return None, None
    if opening:
        start_threshold = low + margin * travel
        low_candidates = np.flatnonzero(values[start:pivot] <= start_threshold)
        search_start = start + int(low_candidates[-1]) if len(low_candidates) else pivot
        motion_start = first_sustained(values, search_start, start_threshold, above=True, frames=sustain_frames)
        motion_end = first_sustained(values, pivot, high - margin * travel, above=True, frames=sustain_frames)
    else:
        search_start = start + int(np.argmax(before))
        motion_start = first_sustained(values, search_start, high - margin * travel, above=False, frames=sustain_frames)
        motion_end = first_sustained(values, pivot, low + margin * travel, above=False, frames=sustain_frames)
    if motion_start is None or motion_end is None or motion_end < motion_start:
        return None, None
    return motion_start, motion_end


def find_events(
    action_gripper: np.ndarray,
    state_gripper: np.ndarray,
    load: np.ndarray,
    *,
    open_threshold: float,
    close_threshold: float,
    sustain_frames: int,
    baseline_frames: int,
    contact_delta: float,
    motion_margin: float,
) -> dict[str, int | float | None]:
    open_index = first_sustained(
        action_gripper, 0, open_threshold, above=True, frames=sustain_frames
    )
    close_index = None if open_index is None else first_sustained(
        action_gripper, open_index + sustain_frames, close_threshold, above=False, frames=sustain_frames
    )
    release_index = None if close_index is None else first_sustained(
        action_gripper, close_index + sustain_frames, open_threshold, above=True, frames=sustain_frames
    )
    reset_index = None if release_index is None else first_sustained(
        action_gripper, release_index + sustain_frames, close_threshold, above=False, frames=sustain_frames
    )
    baseline_stop = min(open_index or len(load), baseline_frames)
    baseline = float(np.median(np.abs(load[:baseline_stop]))) if baseline_stop else 0.0
    contact_index = None if close_index is None else first_sustained(
        np.abs(load), close_index, baseline + contact_delta, above=True, frames=sustain_frames
    )
    actual_close_index = None if close_index is None else first_sustained(
        state_gripper, close_index, close_threshold, above=False, frames=sustain_frames
    )
    close_motion_start = close_motion_end = None
    release_motion_start = release_motion_end = None
    if all(index is not None for index in (open_index, close_index, release_index, reset_index)):
        assert isinstance(open_index, int) and isinstance(close_index, int)
        assert isinstance(release_index, int) and isinstance(reset_index, int)
        close_motion_start, close_motion_end = motion_interval(
            state_gripper, open_index, close_index, release_index, opening=False, sustain_frames=sustain_frames, margin=motion_margin
        )
        release_motion_start, release_motion_end = motion_interval(
            state_gripper, close_index, release_index, reset_index, opening=True, sustain_frames=sustain_frames, margin=motion_margin
        )
    return {
        "open": open_index,
        "close": close_index,
        "contact": contact_index,
        "actual_close": actual_close_index,
        "close_motion_start": close_motion_start,
        "close_motion_end": close_motion_end,
        "release_motion_start": release_motion_start,
        "release_motion_end": release_motion_end,
        "release": release_index,
        "reset_close": reset_index,
        "load_baseline": baseline,
        "contact_threshold": baseline + contact_delta,
    }


def label_episode(
    length: int,
    events: dict[str, int | float | None],
    *,
    hold_frames: int,
    contact_tolerance: int,
) -> tuple[np.ndarray, list[dict[str, int | str]], list[str]]:
    labels = np.full(length, "unknown", dtype=object)
    warnings: list[str] = []
    open_index = events["open"]
    close_index = events["close_motion_start"]
    contact_index = events["contact"]
    release_index = events["release_motion_start"]
    reset_index = events["reset_close"]
    ordered = all(value is not None for value in (open_index, close_index, release_index, reset_index))
    if not ordered:
        return labels, [], ["missing_command_event"]
    assert isinstance(open_index, int) and isinstance(close_index, int)
    assert isinstance(release_index, int) and isinstance(reset_index, int)
    if not (0 <= open_index < close_index < release_index < reset_index <= length):
        return labels, [], ["command_events_out_of_order"]

    labels[:close_index] = "approach"
    labels[release_index:reset_index] = "release"
    command_close = events["close"]
    if not isinstance(contact_index, int) or not isinstance(command_close, int) or abs(contact_index - command_close) > contact_tolerance:
        warnings.append("load_contact_mismatch_diagnostic")
    close_motion_end = events["close_motion_end"]
    grasp_stop = min(release_index, close_motion_end + 1) if isinstance(close_motion_end, int) else min(release_index, close_index + hold_frames)
    if grasp_stop <= close_index:
        warnings.append("empty_grasp_window")
    else:
        labels[close_index:grasp_stop] = "grasp"
        labels[grasp_stop:release_index] = "lift_place"

    intervals: list[dict[str, int | str]] = []
    start = 0
    for index in range(1, length + 1):
        if index == length or labels[index] != labels[start]:
            intervals.append({"start": start, "stop": index, "phase": str(labels[start])})
            start = index
    return labels, intervals, warnings


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(args: argparse.Namespace) -> dict:
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    info_path = dataset / "meta" / "info.json"
    episodes_path = dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    data_path = dataset / "data" / "chunk-000" / "file-000.parquet"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    state_names = info["features"]["observation.state"]["names"]
    action_names = info["features"]["action"]["names"]
    required_states = {"gripper.pos", "gripper.load", "gripper.curr"}
    if not required_states.issubset(state_names) or "gripper.pos" not in action_names:
        raise ValueError("Dataset is missing required gripper telemetry fields")
    fps = float(info["fps"])
    if fps <= 0:
        raise ValueError("Dataset FPS must be positive")

    episodes = pq.read_table(episodes_path).to_pylist()
    data = pq.read_table(data_path, columns=["observation.state", "action", "episode_index", "frame_index"])
    if len(episodes) != info["total_episodes"] or len(data) != info["total_frames"]:
        raise ValueError("Metadata, episode table, and data frame counts disagree")

    state_load = state_names.index("gripper.load")
    state_gripper = state_names.index("gripper.pos")
    action_gripper = action_names.index("gripper.pos")
    phase_rows: list[dict[str, int | str | bool]] = []
    episode_records: dict[str, dict] = {}
    anomaly_episodes: list[int] = []
    phase_counts = {phase: 0 for phase in PHASE_IDS}

    for expected_episode, episode in enumerate(episodes):
        episode_index = int(episode["episode_index"])
        if episode_index != expected_episode:
            raise ValueError("Episode indices must be contiguous and zero-based")
        start = int(episode["dataset_from_index"])
        stop = int(episode["dataset_to_index"])
        length = int(episode["length"])
        if stop - start != length:
            raise ValueError(f"Episode {episode_index} length disagrees with dataset indices")
        state = np.asarray(data["observation.state"].slice(start, length).to_pylist(), dtype=np.float32)
        action = np.asarray(data["action"].slice(start, length).to_pylist(), dtype=np.float32)
        events = find_events(
            action[:, action_gripper],
            state[:, state_gripper],
            state[:, state_load],
            open_threshold=args.open_threshold,
            close_threshold=args.close_threshold,
            sustain_frames=args.sustain_frames,
            baseline_frames=args.baseline_frames,
            contact_delta=args.contact_delta,
            motion_margin=args.motion_margin,
        )
        labels, intervals, warnings = label_episode(
            length,
            events,
            hold_frames=args.hold_frames,
            contact_tolerance=args.contact_tolerance,
        )
        if any(warning != "load_contact_mismatch_diagnostic" for warning in warnings):
            anomaly_episodes.append(episode_index)
        for frame_index, phase in enumerate(labels):
            phase = str(phase)
            phase_counts[phase] += 1
            phase_rows.append(
                {
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "phase_id": PHASE_IDS[phase],
                    "phase": phase,
                    "valid": phase != "unknown",
                    "reviewed": False,
                }
            )
        episode_records[str(episode_index)] = {
            "length": length,
            "events": events,
            "intervals": intervals,
            "warnings": warnings,
            "reviewed": False,
        }

    if len(phase_rows) != info["total_frames"]:
        raise RuntimeError("Exported phase row count disagrees with dataset")
    pq.write_table(pa.Table.from_pylist(phase_rows), output / "phase_labels.parquet")
    annotations = {
        "schema_version": 1,
        "source": "automatic_gripper_telemetry",
        "reviewed": False,
        "training_ready": False,
        "interval_convention": "episode-local [start, stop) frames",
        "phases": PHASE_IDS,
        "config": {
            "open_threshold": args.open_threshold,
            "close_threshold": args.close_threshold,
            "sustain_frames": args.sustain_frames,
            "baseline_frames": args.baseline_frames,
            "contact_delta": args.contact_delta,
            "motion_margin": args.motion_margin,
            "contact_tolerance": args.contact_tolerance,
            "hold_frames": args.hold_frames,
            "hold_seconds": args.hold_frames / fps,
        },
        "dataset": {
            "path": str(dataset),
            "episodes": info["total_episodes"],
            "frames": info["total_frames"],
            "fps": fps,
            "info_sha256": sha256(info_path),
            "episodes_sha256": sha256(episodes_path),
            "data_sha256": sha256(data_path),
        },
        "episodes": episode_records,
    }
    (output / "phase_annotations.json").write_text(
        json.dumps(annotations, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    validation = {
        "status": "partial_pass",
        "training_ready": False,
        "episodes": len(episodes),
        "frames": len(phase_rows),
        "phase_counts": phase_counts,
        "valid_frames": len(phase_rows) - phase_counts["unknown"],
        "unknown_frames": phase_counts["unknown"],
        "anomaly_episodes": anomaly_episodes,
        "gates": {
            "frame_coverage": len(phase_rows) == info["total_frames"],
            "episode_coverage": len(episodes) == info["total_episodes"],
            "ep6_not_rejected_by_load_diagnostic": 6 not in anomaly_episodes,
            "automatic_labels_not_marked_reviewed": all(not row["reviewed"] for row in phase_rows),
        },
        "limitations": [
            "Automatic telemetry labels are not human ground truth.",
            "Unknown/reset frames are excluded from candidate supervision.",
            "Real-robot calibration and force-control validation have not been performed.",
        ],
    }
    if not all(validation["gates"].values()):
        validation["status"] = "failed"
    (output / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--open-threshold", type=float, default=30.0)
    parser.add_argument("--close-threshold", type=float, default=25.0)
    parser.add_argument("--sustain-frames", type=int, default=3)
    parser.add_argument("--baseline-frames", type=int, default=60)
    parser.add_argument("--contact-delta", type=float, default=40.0)
    parser.add_argument("--motion-margin", type=float, default=0.1)
    parser.add_argument("--contact-tolerance", type=int, default=15)
    parser.add_argument("--hold-frames", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    validation = export(parse_args())
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    if validation["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
