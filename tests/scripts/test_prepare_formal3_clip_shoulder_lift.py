import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.prepare_formal3_clip_shoulder_lift import (
    clip_shoulder_lift,
    prepare,
    vector_stats,
)

STATE_NAMES = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos",
    "wrist_roll.pos", "gripper.pos", "gripper.load", "gripper.curr",
]
ACTION_NAMES = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
]


def make_source(root: Path) -> Path:
    source = root / "nomaster"
    (source / "data/chunk-000").mkdir(parents=True)
    (source / "meta/episodes/chunk-000").mkdir(parents=True)
    (source / "annotations").mkdir(parents=True)

    total_frames = 4
    info = {
        "total_episodes": 1,
        "total_frames": total_frames,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [8], "names": STATE_NAMES},
            "action": {"dtype": "float32", "shape": [6], "names": ACTION_NAMES},
        },
    }
    (source / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")

    # shoulder_lift state stays within [-104.4, -95.0]; action overshoots down to -110.
    states = [
        [0.0, -104.4, 70.0, -30.0, 50.0, 20.0, 0.0, 0.0],
        [0.0, -103.0, 70.0, -30.0, 50.0, 20.0, 0.0, 0.0],
        [0.0, -100.0, 70.0, -30.0, 50.0, 20.0, 0.0, 0.0],
        [0.0, -95.0, 70.0, -30.0, 50.0, 20.0, 0.0, 0.0],
    ]
    actions = [
        [0.0, -110.0, 70.0, -30.0, 50.0, 20.0],
        [0.0, -108.8, 70.0, -30.0, 50.0, 20.0],
        [0.0, -105.0, 70.0, -30.0, 50.0, 20.0],
        [0.0, -100.0, 70.0, -30.0, 50.0, 20.0],
    ]
    stats = {
        "observation.state": {
            "min": [0.0, -104.4, 70.0, -30.0, 50.0, 20.0, 0.0, 0.0],
            "max": [0.0, -95.0, 70.0, -30.0, 50.0, 20.0, 0.0, 0.0],
            "mean": [0.0] * 8, "std": [1.0] * 8, "count": [total_frames],
        },
        "action": {
            "min": [0.0, -110.0, 70.0, -30.0, 50.0, 20.0],
            "max": [0.0, -100.0, 70.0, -30.0, 50.0, 20.0],
            "mean": [0.0] * 6, "std": [1.0] * 6, "count": [total_frames],
        },
    }
    (source / "meta/stats.json").write_text(json.dumps(stats), encoding="utf-8")

    schema = pa.schema([
        ("observation.state", pa.list_(pa.float32())),
        ("action", pa.list_(pa.float32())),
        ("timestamp", pa.float64()),
        ("frame_index", pa.int64()),
        ("episode_index", pa.int64()),
        ("index", pa.int64()),
        ("task_index", pa.int64()),
    ])
    rows = [
        {"observation.state": states[i], "action": actions[i],
         "timestamp": float(i) / 30.0, "frame_index": i, "episode_index": 0,
         "index": i, "task_index": 0}
        for i in range(total_frames)
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), source / "data/chunk-000/file-000.parquet")

    episode_rows = [{"episode_index": 0, "length": total_frames}]
    pq.write_table(pa.Table.from_pylist(episode_rows), source / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist([{"task_index": 0, "task": "Cup pick and place"}]),
                   source / "meta/tasks.parquet")
    (source / "annotations/manifest.json").write_text('{"cameras": ["top"]}\n', encoding="utf-8")
    return source


class ClipShoulderLiftTests(unittest.TestCase):
    def test_clip_only_shoulder_lift(self):
        actions = np.array([[1.0, -110.0, 3.0, 4.0, 5.0, 6.0],
                            [1.0, 50.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float64)
        clipped = clip_shoulder_lift(actions, -104.4, 59.5)
        np.testing.assert_array_equal(clipped[:, 1], [-104.4, 50.0])
        np.testing.assert_array_equal(clipped[:, 0], actions[:, 0])
        np.testing.assert_array_equal(clipped[:, 2:], actions[:, 2:])

    def test_vector_stats_min_max(self):
        values = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
        stats = vector_stats(values)
        self.assertEqual(stats["min"], [0.0, 1.0])
        self.assertEqual(stats["max"], [4.0, 5.0])
        self.assertEqual(stats["mean"], [2.0, 3.0])
        self.assertEqual(stats["count"], [3])

    def test_prepare_clips_action_and_recomputes_stats(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            root = Path(folder)
            source = make_source(root)
            destination = root / "clipsh"
            output = root / "record.json"
            record = prepare(source, destination, output, None, None)

            self.assertEqual(record["status"], "passed")
            clip = record["clip"]
            self.assertEqual(clip["lower"], -104.4)
            self.assertEqual(clip["upper"], -95.0)
            self.assertEqual(clip["frames_below_lower"], 3)
            self.assertEqual(clip["frames_above_upper"], 0)

            written = np.asarray(
                pq.read_table(destination / "data/chunk-000/file-000.parquet")["action"].to_pylist(),
                dtype=np.float32,
            )
            self.assertAlmostEqual(float(np.min(written[:, 1])), -104.4, delta=1e-3)

            new_stats = json.loads((destination / "meta/stats.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(new_stats["action"]["min"][1], -104.4, delta=1e-3)

    def test_prepare_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            root = Path(folder)
            source = make_source(root)
            destination = root / "clipsh"
            output = root / "record.json"
            prepare(source, destination, output, None, None)
            with self.assertRaises(FileExistsError):
                prepare(source, destination, root / "record2.json", None, None)


if __name__ == "__main__":
    unittest.main()
