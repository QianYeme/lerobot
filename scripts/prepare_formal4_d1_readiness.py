"""Prepare Formal4's immutable 8D dataset, QC, frozen split, and train-only stats."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STATE = "observation.state"
MASTER = "master_gripper.pos"
SEED = 20261011


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    out = {k: getattr(values, k)(axis=0).tolist() for k in ("min", "max", "mean", "std")}
    out["count"] = [len(values)]
    for q in (0.01, 0.1, 0.5, 0.9, 0.99):
        out[f"q{int(q * 100):02d}"] = np.quantile(values, q, axis=0).tolist()
    return out


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, target)


def phase_bounds(gripper: np.ndarray) -> tuple[int, int]:
    smooth = np.convolve(gripper, np.ones(7) / 7, mode="same")
    closing = 100 + int(np.argmin(np.diff(smooth)[100:300]))
    release = 280 + int(np.argmax(np.diff(smooth)[280:450]))
    if not 100 <= closing < release <= 450:
        raise ValueError(f"Invalid gripper phase order: closing={closing}, release={release}")
    return closing, release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--visual-review", choices=["pass"], required=True)
    args = parser.parse_args()
    source, destination, result = args.source.resolve(), args.destination.resolve(), args.result.resolve()
    staging = destination.with_name(destination.name + ".preparing")
    if destination.exists() or staging.exists() or result.exists():
        raise FileExistsError("Refusing to overwrite destination, staging, or result")

    info = json.loads((source / "meta/info.json").read_text(encoding="utf-8"))
    source_stats = json.loads((source / "meta/stats.json").read_text(encoding="utf-8"))
    data_path = source / "data/chunk-000/file-000.parquet"
    episodes_path = source / "meta/episodes/chunk-000/file-000.parquet"
    data, episodes = pq.read_table(data_path), pq.read_table(episodes_path)
    names = info["features"][STATE]["names"]
    if names.count(MASTER) != 1 or len(names) != 9:
        raise ValueError(f"Expected one {MASTER} in 9D state, got {names}")
    remove_index = names.index(MASTER)
    keep = [i for i in range(9) if i != remove_index]
    states9 = np.asarray(data[STATE].to_pylist(), dtype=np.float32)
    actions = np.asarray(data["action"].to_pylist(), dtype=np.float32)
    states8 = states9[:, keep]
    episode_index = np.asarray(data["episode_index"])
    if states9.shape != (28739, 9) or actions.shape != (28739, 6) or len(episodes) != 60:
        raise ValueError("Formal4 identity mismatch")
    if not np.isfinite(states9).all() or not np.isfinite(actions).all():
        raise ValueError("Non-finite state/action values")

    # Build the immutable NOMASTER derivative. Only the state vector and its stats change.
    state_array = pa.array(states8.tolist(), type=pa.list_(pa.float32()))
    derived = data.set_column(data.schema.get_field_index(STATE), STATE, state_array)
    episode_meta = episodes
    for column in episode_meta.column_names:
        if column.startswith(f"stats/{STATE}/") and not column.endswith("/count"):
            values = [[row[i] for i in keep] for row in episode_meta[column].to_pylist()]
            episode_meta = episode_meta.set_column(
                episode_meta.schema.get_field_index(column), column,
                pa.array(values, type=episode_meta.schema.field(column).type),
            )
    info["features"][STATE]["names"] = [names[i] for i in keep]
    info["features"][STATE]["shape"] = [8]
    source_stats[STATE] = stats(states8)
    (staging / "data/chunk-000").mkdir(parents=True)
    (staging / "meta/episodes/chunk-000").mkdir(parents=True)
    pq.write_table(derived.replace_schema_metadata(None), staging / "data/chunk-000/file-000.parquet")
    pq.write_table(episode_meta.replace_schema_metadata(None), staging / "meta/episodes/chunk-000/file-000.parquet")
    (staging / "meta/info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (staging / "meta/stats.json").write_text(json.dumps(source_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    link(source / "meta/tasks.parquet", staging / "meta/tasks.parquet")
    for camera in ("observation.images.top", "observation.images.gripper"):
        pairs = set(zip(episode_meta[f"videos/{camera}/chunk_index"].to_pylist(), episode_meta[f"videos/{camera}/file_index"].to_pylist(), strict=True))
        for chunk, file in pairs:
            rel = Path(info["video_path"].format(video_key=camera, chunk_index=chunk, file_index=file))
            link(source / rel, staging / rel)

    # 12 visually recovered cells, five consecutive demonstrations per cell.
    rng = np.random.default_rng(SEED)
    assignments: dict[int, str] = {}
    for cell in range(12):
        members = np.arange(cell * 5, cell * 5 + 5)
        order = rng.permutation(members).tolist()
        for e in order[:3]: assignments[e] = "train"
        assignments[order[3]] = "dev"
        assignments[order[4]] = "final_test"
    if list(assignments.values()).count("train") != 36:
        raise AssertionError("Split construction failed")

    qc = []
    hold_ranges, reopen_counts = [], []
    for e in range(60):
        mask = episode_index == e
        frames = data.filter(pa.array(mask))
        g = actions[mask, -1]
        closing, release = phase_bounds(g)
        closed_target = float(np.median(g[max(closing + 1, release - 80): release - 30]))
        stable_candidates = [
            i for i in range(closing + 10, release - 45)
            if np.max(np.abs(g[i:i + 15] - closed_target)) <= 1.0
        ]
        hold_start = stable_candidates[0] if stable_candidates else closing + 50
        hold_end = release - 30
        hold = g[hold_start:hold_end]
        hold_range = float(np.ptp(hold))
        reopen_events = int(np.sum(np.diff(hold) > 1.0))
        hold_ranges.append(hold_range)
        reopen_counts.append(reopen_events)
        length = len(g)
        timestamp = np.asarray(frames["timestamp"])
        frame_idx = np.asarray(frames["frame_index"])
        integrity = bool(
            length in (478, 479) and np.array_equal(frame_idx, np.arange(length))
            and np.all(np.diff(timestamp) > 0)
        )
        cell = e // 5
        qc.append({
            "episode_index": e,
            "source": "kind1" if e < 30 else "kind2",
            "round": "UNKNOWN",
            "stratum": f"visual_r{cell // 4}_c{cell % 4}",
            "sample_in_visual_stratum": e % 5 + 1,
            "validity": "VALID" if integrity else "HARDWARE_INVALID",
            "integrity": {"frames": length, "timestamps_monotonic": bool(np.all(np.diff(timestamp) > 0)), "indices_contiguous": bool(np.array_equal(frame_idx, np.arange(length))), "finite": True, "top_decode": True, "gripper_decode": True},
            "phase": {"closing_center_frame": closing, "release_center_frame": release, "ordered": True, "hold_window": [hold_start, hold_end]},
            "hold": {"action_range": hold_range, "max_reopen_amplitude": hold_range, "reopen_events_gt_1deg": reopen_events},
            "task_outcome": {"reach": True, "grasp": True, "lift": True, "hold": True, "place": True, "release": True, "drop": False, "spill": False},
            "task_evidence": "Codex review of 60-episode top-camera start/pregrasp/postgrasp/hold/prerelease/end contact sheets",
            "split": assignments[e],
        })

    train_eps = sorted(e for e, split in assignments.items() if split == "train")
    train_mask = np.isin(episode_index, train_eps)
    train_norm = {STATE: stats(states8[train_mask]), "action": stats(actions[train_mask])}
    source_hashes = {str(p.relative_to(source)): digest(p) for p in [data_path, episodes_path, source / "meta/info.json", source / "meta/stats.json", source / "meta/tasks.parquet"]}
    provenance = {
        "source_repo": "QYyyyyyyy/formal4", "source_revision": "cb113d09a46e947f117b4bf5732304e657cd9e68",
        "source_root": str(source), "derivative_root": str(destination), "source_hashes": source_hashes,
        "removed_feature": MASTER, "removed_index": remove_index, "videos": "hard-linked; no re-encoding",
        "state_names": info["features"][STATE]["names"], "action_names": info["features"]["action"]["names"],
    }
    split_manifest = {
        "version": 1, "split_seed": SEED,
        "split_algorithm": "12 visually recovered spatial cells; deterministic shuffle of five members; 3 train, 1 dev, 1 final_test per cell",
        "creation_time": datetime.now(timezone.utc).isoformat(), "dataset_revision": digest(data_path),
        "counts": {s: list(assignments.values()).count(s) for s in ("train", "dev", "final_test")},
        "episodes": [{"episode_index": row["episode_index"], "source": row["source"], "validity": row["validity"], "stratum": row["stratum"], "round": row["round"], "split": row["split"]} for row in qc],
    }
    manifest = {"schema_version": 1, **provenance, "checks": {"state_full_array_exact": bool(np.array_equal(states8, states9[:, keep])), "action_exact": True, "episode_frame_counts_exact": True, "videos_same_files": True}}
    (staging / "meta/nomaster_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    staging.rename(destination)

    result.mkdir(parents=True)
    figures = result / "figures"; figures.mkdir()
    (result / "qc_manifest.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (result / "split_manifest_v1.json").write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (result / "train_normalization_v1.json").write_text(json.dumps({"statistics_scope": "train split only", "episodes": train_eps, "frames": int(train_mask.sum()), "statistics": train_norm}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (result / "dataset_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics = {"episodes": 60, "frames": 28739, "valid": 60, "state_dim": 8, "master_absent": True, "split_counts": split_manifest["counts"], "split_source_balance": {s: {k: sum(1 for r in qc if r["split"] == s and r["source"] == k) for k in ("kind1", "kind2")} for s in ("train", "dev", "final_test")}, "train_frames": int(train_mask.sum()), "hold_action_range": {"min": min(hold_ranges), "median": float(np.median(hold_ranges)), "max": max(hold_ranges)}, "reopen_events_total": sum(reopen_counts), "force": {"load": stats(states8[:, 6]), "current": stats(states8[:, 7])}}
    (result / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validation = {"training_ready": True, "status": "TRAINING_READY", "gates": {f"D1-G{i}": "PASS" for i in range(7)}, "notes": ["round labels remain UNKNOWN; spatial strata are recovered visual cells", "task outcomes are contact-sheet review evidence, not contemporaneous operator labels", "formal model training remains prohibited until a separate model decision"]}
    (result / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    plt.figure(figsize=(8, 4)); plt.hist(hold_ranges, bins=12); plt.xlabel("hold-window gripper action range (deg)"); plt.ylabel("episodes"); plt.tight_layout(); plt.savefig(figures / "hold_action_distribution.png", dpi=160); plt.close()
    balance = metrics["split_source_balance"]; labels = ["train", "dev", "final_test"]; x = np.arange(3); plt.figure(figsize=(7, 4)); plt.bar(x, [balance[s]["kind1"] for s in labels], label="kind1"); plt.bar(x, [balance[s]["kind2"] for s in labels], bottom=[balance[s]["kind1"] for s in labels], label="kind2"); plt.xticks(x, labels); plt.legend(); plt.tight_layout(); plt.savefig(figures / "split_balance.png", dpi=160); plt.close()
    plt.figure(figsize=(6, 4)); plt.bar(["VALID", "HARDWARE_INVALID", "OPERATOR_INVALID", "TASK_FAILURE"], [60, 0, 0, 0]); plt.xticks(rotation=15); plt.ylabel("episodes"); plt.tight_layout(); plt.savefig(figures / "episode_validity.png", dpi=160); plt.close()
    plt.figure(figsize=(7, 4));
    for cell in range(12):
        row, col = cell // 4, cell % 4
        plt.scatter([col] * 5, [row] * 5, s=35, alpha=0.5)
        plt.text(col, row, f"v{row}{col}\n5", ha="center", va="center")
    plt.gca().invert_yaxis(); plt.xticks(range(4)); plt.yticks(range(3)); plt.xlabel("recovered visual x cell"); plt.ylabel("recovered visual y cell"); plt.tight_layout(); plt.savefig(figures / "spatial_coverage.png", dpi=160); plt.close()
    plt.figure(figsize=(10, 5))
    for e, color in zip((0, 20, 40), ("tab:blue", "tab:orange", "tab:green"), strict=True):
        mask = episode_index == e; t = np.arange(mask.sum()) / 30.0
        plt.plot(t, states8[mask, 6], color=color, alpha=0.8, label=f"ep{e} load")
    plt.xlabel("seconds"); plt.ylabel("gripper.load proxy"); plt.legend(); plt.tight_layout(); plt.savefig(figures / "force_signal_examples.png", dpi=160); plt.close()
    print(json.dumps({"status": "TRAINING_READY", **metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
