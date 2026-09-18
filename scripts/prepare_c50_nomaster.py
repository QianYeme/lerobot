"""Create an immutable C50 derivative: 8 observable states, train-only statistics."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.factory import IMAGENET_STATS
from lerobot.datasets.utils import unflatten_dict
from lerobot.policies.act_det.label_loader import LabelLoader

VALIDATION = [7, 11, 13, 14, 16, 27, 35, 38, 40, 43]
TRAINING = [ep for ep in range(50) if ep not in VALIDATION]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def vector_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    result = {"min": values.min(0), "max": values.max(0), "mean": values.mean(0),
              "std": values.std(0), "count": np.array([len(values)])}
    result.update({f"q{int(q * 100):02d}": np.quantile(values, q, axis=0)
                   for q in [0.01, 0.1, 0.5, 0.9, 0.99]})
    return {key: value.tolist() for key, value in result.items()}


def prepare(source, destination):
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    info = json.loads((source / "meta/info.json").read_text(encoding="utf-8"))
    names = info["features"]["observation.state"]["names"]
    assert len(names) == 9 and names[-1] == "master_gripper.pos"
    data = pa.concat_tables([pq.read_table(path) for path in sorted((source / "data").glob("*/*.parquet"))])
    data = data.filter(pa.compute.less(data["episode_index"], 50))
    episodes = pa.concat_tables([pq.read_table(path) for path in sorted((source / "meta/episodes").glob("*/*.parquet"))])
    episodes = episodes.filter(pa.compute.less(episodes["episode_index"], 50))
    assert len(data) == 35900 and len(episodes) == 50
    assert data["index"].to_pylist() == list(range(35900))
    state_original = np.asarray(data["observation.state"].to_pylist(), dtype=np.float32)
    state = state_original[:, :8].copy()
    action = np.asarray(data["action"].to_pylist(), dtype=np.float32)
    assert np.isfinite(state).all() and np.isfinite(action).all()
    assert np.array_equal(state_original[:, 8], action[:, 5])
    labels = LabelLoader(str(source / "annotations"), {"observation.images.top": "top"})
    for ep in range(50):
        frames = data.filter(pa.compute.equal(data["episode_index"], ep))["frame_index"].to_pylist()
        assert frames == list(range(718))
        row = episodes.filter(pa.compute.equal(episodes["episode_index"], ep)).to_pylist()[0]
        assert row["dataset_from_index"] == ep * 718 and row["dataset_to_index"] == (ep + 1) * 718
        assert row["data/chunk_index"] == row["data/file_index"] == 0
        for frame in frames:
            label = labels.get_labels("observation.images.top", ep, frame)
            assert label and label["labels"] == ["cup"], (ep, frame, label)
            for x1, y1, x2, y2 in label["bboxes"]:
                assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480, (ep, frame)
    source_hashes = {str(path.relative_to(source)): digest(path) for path in
                     [source / "meta/info.json", source / "meta/stats.json", *sorted((source / "data").glob("*/*.parquet"))]}
    data = data.set_column(data.schema.get_field_index("observation.state"), "observation.state",
                           pa.array(state.tolist(), type=pa.list_(pa.float32())))
    for column in episodes.column_names:
        if column.startswith("stats/observation.state/") and not column.endswith("/count"):
            episodes = episodes.set_column(episodes.schema.get_field_index(column), column,
                                           pa.array([value[:8] for value in episodes[column].to_pylist()],
                                                    type=episodes.schema.field(column).type))
    training_mask = np.isin(data["episode_index"].to_numpy(), TRAINING)
    assert training_mask.sum() == 28720
    stats = {column: vector_stats(np.asarray(data[column].to_pylist())[training_mask])
             for column in data.column_names}
    training_rows = [row for row in episodes.to_pylist() if row["episode_index"] in TRAINING]
    camera_stats = aggregate_stats([unflatten_dict({key.removeprefix("stats/"): np.asarray(value)
                                    for key, value in row.items() if key.startswith("stats/observation.images.")})
                                   for row in training_rows])
    for camera, values in camera_stats.items():
        stats[camera] = {key: value.tolist() for key, value in values.items()}
        stats[camera].update(IMAGENET_STATS)
    tasks = pq.read_table(source / "meta/tasks.parquet")
    tasks = tasks.filter(pa.compute.is_in(tasks["task_index"], value_set=pa.array(sorted(set(data["task_index"].to_pylist())))))
    info.update(total_episodes=50, total_frames=len(data), total_tasks=len(tasks), splits={"train": "0:50"})
    info["features"]["observation.state"].update(names=names[:8], shape=[8])
    for relative in ["data/chunk-000", "meta/episodes/chunk-000", "annotations/top"]:
        (destination / relative).mkdir(parents=True, exist_ok=False)
    pq.write_table(data.replace_schema_metadata(None), destination / "data/chunk-000/file-000.parquet")
    pq.write_table(episodes.replace_schema_metadata(None), destination / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(tasks, destination / "meta/tasks.parquet")
    for camera in camera_stats:
        relative = Path(info["video_path"].format(video_key=camera, chunk_index=0, file_index=0))
        assert all(row[f"videos/{camera}/file_index"] == row[f"videos/{camera}/chunk_index"] == 0 for row in episodes.to_pylist())
        (destination / relative).parent.mkdir(parents=True, exist_ok=True)
        os.link(source / relative, destination / relative)
        assert os.path.samefile(source / relative, destination / relative)
    annotation_hashes = {}
    for ep in range(50):
        relative = Path(f"annotations/top/episode_{ep:03d}.xml")
        shutil.copy2(source / relative, destination / relative)
        annotation_hashes[str(relative)] = digest(destination / relative)
        assert annotation_hashes[str(relative)] == digest(source / relative)
    for filename, value in [("info.json", info), ("stats.json", stats)]:
        (destination / "meta" / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    reloaded = pq.read_table(destination / "data/chunk-000/file-000.parquet")
    assert np.array_equal(np.asarray(reloaded["observation.state"].to_pylist()), state)
    for column in data.column_names:
        if column != "observation.state":
            assert reloaded[column].equals(data[column]), column
    assert all(digest(source / relative) == value for relative, value in source_hashes.items())
    manifest = {"source": str(source.resolve()), "destination": str(destination.resolve()),
                "removed_feature": "master_gripper.pos", "state_names": names[:8], "action_names": info["features"]["action"]["names"],
                "train": TRAINING, "validation": VALIDATION, "train_frames": 28720, "validation_frames": 7180,
                "statistics_episodes": TRAINING, "camera_normalization": "ImageNet constants; other camera statistics aggregated from training episodes",
                "source_hashes": source_hashes, "annotation_sha256": annotation_hashes,
                "videos": "hard-linked original file-000.mp4, read-only reuse, no re-encoding",
                "checks": "35900 unique frames; 8 finite state dims; unchanged action/time/index; 35900 valid top cup labels; source unchanged"}
    (destination / "meta/nomaster_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source, args.destination)
