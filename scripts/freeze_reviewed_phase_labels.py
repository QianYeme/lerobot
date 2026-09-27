"""Freeze reviewed formal3 event boundaries into per-frame phase labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PHASE_IDS = {"unknown": -1, "approach": 0, "grasp": 1, "lift_place": 2, "release": 3}
BOUNDARY_COLUMNS = (
    "closing_start_frame",
    "closing_end_frame",
    "lift_start_frame",
    "release_start_frame",
    "release_end_frame",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def boundaries(review: dict) -> tuple[int, int, int, int, int]:
    steps = review["steps"]
    return (
        int(steps["closing"]["start_frame"]),
        int(steps["closing"]["end_frame"]),
        int(steps["lift_start"]["frame"]),
        int(steps["release"]["start_frame"]),
        int(steps["release"]["end_frame"]),
    )


def labels_from_review(length: int, review: dict) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    close_start, close_end, lift_start, release_start, release_end = boundaries(review)
    if not review.get("reviewed"):
        raise ValueError("Episode is not reviewed")
    if not (0 <= close_start <= close_end < lift_start <= release_start <= release_end < length):
        raise ValueError("Reviewed boundaries are out of order or outside the episode")

    labels = np.full(length, "unknown", dtype=object)
    labels[:close_start] = "approach"
    labels[close_start:lift_start] = "grasp"
    labels[lift_start:release_start] = "lift_place"
    labels[release_start : release_end + 1] = "release"
    intervals = [
        {"start": 0, "stop": close_start, "phase": "approach"},
        {"start": close_start, "stop": lift_start, "phase": "grasp"},
        {"start": lift_start, "stop": release_start, "phase": "lift_place"},
        {"start": release_start, "stop": release_end + 1, "phase": "release"},
        {"start": release_end + 1, "stop": length, "phase": "unknown"},
    ]
    return labels, [interval for interval in intervals if interval["start"] < interval["stop"]]


def validate_csv(json_episodes: dict[str, dict], csv_path: Path) -> None:
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(json_episodes):
        raise ValueError("Review CSV and JSON episode counts disagree")
    for row in rows:
        episode = row["episode_index"]
        if episode not in json_episodes:
            raise ValueError(f"Review CSV contains unknown episode {episode}")
        csv_boundaries = tuple(int(row[column]) for column in BOUNDARY_COLUMNS)
        if csv_boundaries != boundaries(json_episodes[episode]):
            raise ValueError(f"Review CSV and JSON boundaries disagree for episode {episode}")
        if row["reviewed"].lower() != "true" or not json_episodes[episode].get("reviewed"):
            raise ValueError(f"Episode {episode} is not reviewed in both files")


def freeze(args: argparse.Namespace) -> dict:
    dataset = args.dataset.resolve()
    reviews_path = args.reviews.resolve()
    review_csv_path = args.review_csv.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    info_path = dataset / "meta" / "info.json"
    episodes_path = dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    data_path = dataset / "data" / "chunk-000" / "file-000.parquet"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    dataset_episodes = pq.read_table(episodes_path).to_pylist()
    reviews_document = json.loads(reviews_path.read_text(encoding="utf-8"))
    if reviews_document.get("schema_version") != 3:
        raise ValueError("Expected review schema_version 3")
    reviews = reviews_document["episodes"]
    if len(reviews) != info["total_episodes"] or len(dataset_episodes) != info["total_episodes"]:
        raise ValueError("Dataset and review episode counts disagree")
    validate_csv(reviews, review_csv_path)

    rows: list[dict[str, int | str | bool]] = []
    frozen_episodes: dict[str, dict] = {}
    phase_counts = {phase: 0 for phase in PHASE_IDS}
    hold_frames: list[int] = []
    for expected_episode, metadata in enumerate(dataset_episodes):
        episode = int(metadata["episode_index"])
        if episode != expected_episode or str(episode) not in reviews:
            raise ValueError("Episode indices must be contiguous and present in reviews")
        length = int(metadata["length"])
        review = reviews[str(episode)]
        if int(review["episode_index"]) != episode or int(review["length"]) != length:
            raise ValueError(f"Episode {episode} identity or length disagrees")
        labels, intervals = labels_from_review(length, review)
        close_start, close_end, lift_start, release_start, release_end = boundaries(review)
        hold_frames.append(lift_start - close_end)
        for frame, phase_value in enumerate(labels):
            phase = str(phase_value)
            phase_counts[phase] += 1
            rows.append(
                {
                    "episode_index": episode,
                    "frame_index": frame,
                    "phase_id": PHASE_IDS[phase],
                    "phase": phase,
                    "valid": phase != "unknown",
                    "reviewed": True,
                }
            )
        frozen_episodes[str(episode)] = {
            "length": length,
            "boundaries": {
                "closing_start": close_start,
                "closing_end": close_end,
                "lift_start": lift_start,
                "release_start": release_start,
                "release_end": release_end,
            },
            "hold_frames": lift_start - close_end,
            "intervals": intervals,
            "reviewed": True,
        }

    frame_count = len(rows)
    gates = {
        "episode_coverage": len(frozen_episodes) == info["total_episodes"],
        "frame_coverage": frame_count == info["total_frames"],
        "all_episodes_reviewed": all(record["reviewed"] for record in frozen_episodes.values()),
        "all_rows_reviewed": all(row["reviewed"] for row in rows),
        "all_phase_intervals_nonempty": all(
            all(interval["start"] < interval["stop"] for interval in record["intervals"])
            for record in frozen_episodes.values()
        ),
    }
    training_ready = all(gates.values())
    pq.write_table(pa.Table.from_pylist(rows), output / "reviewed_phase_labels.parquet")
    annotations = {
        "schema_version": 1,
        "source": "human_reviewed_event_boundaries",
        "training_ready": training_ready,
        "interval_convention": "episode-local [start, stop) frames",
        "release_end_input_is_inclusive": True,
        "phases": PHASE_IDS,
        "dataset": {
            "path": str(dataset),
            "episodes": info["total_episodes"],
            "frames": info["total_frames"],
            "fps": info["fps"],
        },
        "episodes": frozen_episodes,
    }
    (output / "reviewed_phase_annotations.json").write_text(
        json.dumps(annotations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    validation = {
        "status": "pass" if training_ready else "failed",
        "training_ready": training_ready,
        "episodes": len(frozen_episodes),
        "frames": frame_count,
        "phase_counts": phase_counts,
        "valid_frames": frame_count - phase_counts["unknown"],
        "unknown_frames": phase_counts["unknown"],
        "hold_frames": {
            "min": min(hold_frames),
            "median": float(np.median(hold_frames)),
            "max": max(hold_frames),
        },
        "gates": gates,
    }
    (output / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "dataset_info_sha256": sha256(info_path),
        "dataset_episodes_sha256": sha256(episodes_path),
        "dataset_data_sha256": sha256(data_path),
        "review_json_sha256": sha256(reviews_path),
        "review_csv_sha256": sha256(review_csv_path),
        "reviewed_phase_labels_sha256": sha256(output / "reviewed_phase_labels.parquet"),
        "reviewed_phase_annotations_sha256": sha256(output / "reviewed_phase_annotations.json"),
    }
    (output / "freeze_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return validation


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "数据集/formal3/kind_merged")
    parser.add_argument("--reviews", type=Path, default=root / "outputs/formal3_phase_labels_20260927/review_ui_v3_reviewed/phase_step_reviews.json")
    parser.add_argument("--review-csv", type=Path, default=root / "outputs/formal3_phase_labels_20260927/review_ui_v3_reviewed/phase_step_reviews.csv")
    parser.add_argument("--output", type=Path, default=root / "outputs/formal3_phase_labels_20260927/final_reviewed_labels")
    return parser.parse_args()


def main() -> None:
    validation = freeze(parse_args())
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if not validation["training_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
