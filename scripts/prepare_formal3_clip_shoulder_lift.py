"""Clip the shoulder_lift action label to the follower's observed reachable range.

Background (2026-09-28 real-robot diagnosis):
  The four Formal3 models freeze at the grasp phase when a cup is present: the
  model commands shoulder_lift to ~-110 deg (learned from the leader's action
  min of -108.84 deg) while the follower physically meets the cup at -104.44 deg
  and cannot descend further, so the model never "arrives" and never closes the
  gripper. The leader (action label) is a separate arm whose joint mapping is
  offset from the follower (state), so the grasp-depth command overshoots the
  follower's reachable range. Clipping the shoulder_lift action to the
  follower's observed state range removes that leader/follower mismatch.

This script produces a new NOMASTER derivative `kind_merged_nomaster_clipsh`
whose `action.shoulder_lift.pos` is clipped to [lo, hi] (defaults to the
source state's per-joint min/max). State, frames, detection annotations and
videos are untouched and hard-linked. Action statistics are recomputed over all
frames; state statistics are carried over unchanged.

Run it on the machine that holds the full dataset (the training server), then
derive the train-only fit48 view with the existing
`prepare_formal3_fit48_dataset.py` before retraining.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


STATE_KEY = "observation.state"
ACTION_KEY = "action"
SHOULDER_LIFT = "shoulder_lift.pos"


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


def hardlink_tree(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if path.is_dir():
            (destination / relative).mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            (destination / relative).parent.mkdir(parents=True, exist_ok=True)
            os.link(path, destination / relative)


def clip_shoulder_lift(actions: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip the shoulder_lift column (index 1) of a 6-D action array to [lo, hi]."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 6:
        raise ValueError(f"Expected (N, 6) action array, got {actions.shape}")
    clipped = actions.copy()
    clipped[:, 1] = np.clip(clipped[:, 1], lo, hi)
    return clipped


def prepare(source: Path, destination: Path, output: Path,
            shoulder_lift_min: float | None, shoulder_lift_max: float | None) -> dict:
    source = source.resolve()
    destination = destination.resolve()
    output = output.resolve()
    staging = destination.with_name(f"{destination.name}.preparing")
    if destination.exists() or staging.exists():
        raise FileExistsError(f"Refusing to overwrite {destination} or {staging}")

    info_path = source / "meta/info.json"
    stats_path = source / "meta/stats.json"
    data_path = source / "data/chunk-000/file-000.parquet"
    if not (info_path.is_file() and stats_path.is_file() and data_path.is_file()):
        raise FileNotFoundError("Source info.json / stats.json / data parquet are missing")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    state_names = info["features"][STATE_KEY]["names"]
    action_names = info["features"][ACTION_KEY]["names"]
    if len(state_names) != 8 or "master_gripper.pos" in state_names:
        raise ValueError(f"Source is not the 8-D NOMASTER dataset: {state_names}")
    if action_names != [
        "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
        "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
    ]:
        raise ValueError(f"Unexpected action schema: {action_names}")
    shoulder_lift_index = action_names.index(SHOULDER_LIFT)
    if shoulder_lift_index != 1:
        raise ValueError(f"shoulder_lift.pos is not action dimension 1: {action_names}")

    data = pq.read_table(data_path)
    states = np.asarray(data[STATE_KEY].to_pylist(), dtype=np.float64)
    actions = np.asarray(data[ACTION_KEY].to_pylist(), dtype=np.float64)
    if states.shape != (info["total_frames"], 8) or actions.shape != (info["total_frames"], 6):
        raise ValueError("Source state/action dimensions disagree with metadata")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Source contains non-finite state or action values")

    state_min = stats[STATE_KEY]["min"]
    state_max = stats[STATE_KEY]["max"]
    lo = float(shoulder_lift_min) if shoulder_lift_min is not None else float(state_min[shoulder_lift_index])
    hi = float(shoulder_lift_max) if shoulder_lift_max is not None else float(state_max[shoulder_lift_index])
    if lo > hi:
        raise ValueError(f"Clip lower bound {lo} exceeds upper bound {hi}")

    clipped_actions = clip_shoulder_lift(actions, lo, hi)
    n_below = int((actions[:, shoulder_lift_index] < lo).sum())
    n_above = int((actions[:, shoulder_lift_index] > hi).sum())
    if n_below == 0 and n_above == 0:
        raise ValueError("Clip is a no-op: no shoulder_lift action lies outside the clip range")

    action_type = data.schema.field(ACTION_KEY).type
    write_dtype = action_type.value_type.to_pandas_dtype()
    written_actions = clipped_actions.astype(write_dtype)
    data_clipped = data.set_column(
        data.schema.get_field_index(ACTION_KEY),
        ACTION_KEY,
        pa.array(written_actions.tolist(), type=action_type),
    )

    staging_data = staging / "data/chunk-000"
    staging_data.mkdir(parents=True)
    pq.write_table(data_clipped.replace_schema_metadata(None), staging_data / "file-000.parquet")

    (staging / "meta").mkdir(parents=True)
    hardlink_tree(source / "meta/episodes", staging / "meta/episodes")
    for entry in ("info.json", "tasks.parquet"):
        os.link(source / "meta" / entry, staging / "meta" / entry)
    for entry in ("videos", "annotations"):
        if (source / entry).exists():
            hardlink_tree(source / entry, staging / entry)

    new_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    new_stats[ACTION_KEY] = vector_stats(written_actions)
    (staging / "meta/stats.json").write_text(
        json.dumps(new_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "schema_version": 1,
        "source": str(source),
        "destination": str(destination),
        "clip": {
            "feature": f"{ACTION_KEY}/{SHOULDER_LIFT}",
            "index": shoulder_lift_index,
            "lower": lo,
            "upper": hi,
            "bound_source": "source state stats min/max (follower observed range)"
            if shoulder_lift_min is None and shoulder_lift_max is None
            else "explicit CLI override",
            "frames_below_lower": n_below,
            "frames_above_upper": n_above,
            "action_shoulder_lift_min_before": float(np.min(actions[:, shoulder_lift_index])),
            "action_shoulder_lift_max_before": float(np.max(actions[:, shoulder_lift_index])),
            "action_shoulder_lift_min_after": float(np.min(written_actions[:, shoulder_lift_index])),
            "action_shoulder_lift_max_after": float(np.max(written_actions[:, shoulder_lift_index])),
        },
        "episodes": info["total_episodes"],
        "frames": info["total_frames"],
        "statistics_scope": "action recomputed over all frames after clipping; state carried over unchanged",
        "checks": {
            "state_8d": True,
            "action_6d": True,
            "non_action_data_columns_exact": True,
            "action_clipped": n_below + n_above > 0,
        },
    }
    manifest_path = staging / "meta/clip_shoulder_lift_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    written = pq.read_table(staging_data / "file-000.parquet")
    written_back = np.asarray(written[ACTION_KEY].to_pylist(), dtype=np.float64)
    if not np.array_equal(written_back, written_actions.astype(np.float64)):
        raise AssertionError("Written clipped actions differ from expected values")
    for column in data.column_names:
        if column != ACTION_KEY and not written[column].equals(data[column]):
            raise AssertionError(f"Unexpected changed data column: {column}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(destination)

    record = {
        "status": "passed",
        **manifest,
        "stats_sha256": sha256(destination / "meta/stats.json"),
        "manifest_sha256": sha256(destination / "meta/clip_shoulder_lift_manifest.json"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shoulder-lift-min", type=float, default=None,
                        help="Clip lower bound in degrees (default: source state min)")
    parser.add_argument("--shoulder-lift-max", type=float, default=None,
                        help="Clip upper bound in degrees (default: source state max)")
    args = parser.parse_args()
    record = prepare(
        args.source, args.destination, args.output,
        args.shoulder_lift_min, args.shoulder_lift_max,
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
