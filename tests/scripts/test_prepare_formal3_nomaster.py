import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.prepare_formal3_nomaster import prepare


class PrepareFormal3NomasterTests(unittest.TestCase):
    def test_only_master_state_is_removed(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            root = Path(folder)
            source = root / "source"
            destination = root / "nomaster"
            (source / "data/chunk-000").mkdir(parents=True)
            (source / "meta/episodes/chunk-000").mkdir(parents=True)
            names = [
                "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos",
                "wrist_roll.pos", "gripper.pos", "gripper.load", "gripper.curr", "master_gripper.pos",
            ]
            info = {
                "total_episodes": 1,
                "total_frames": 2,
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [9], "names": names},
                    "action": {"dtype": "float32", "shape": [6], "names": names[:6]},
                    "observation.images.top": {"dtype": "video", "shape": [2, 2, 3], "names": None},
                },
            }
            (source / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
            stats = {"observation.state": {"mean": [0.0] * 9}, "action": {"mean": [0.0] * 6}}
            (source / "meta/stats.json").write_text(json.dumps(stats), encoding="utf-8")
            data = pa.Table.from_pylist([
                {"observation.state": [float(i) for i in range(9)], "action": [1.0] * 6,
                 "timestamp": 0.0, "frame_index": 0, "episode_index": 0, "index": 0, "task_index": 0},
                {"observation.state": [float(i + 1) for i in range(9)], "action": [2.0] * 6,
                 "timestamp": 1.0, "frame_index": 1, "episode_index": 0, "index": 1, "task_index": 0},
            ])
            pq.write_table(data, source / "data/chunk-000/file-000.parquet")
            episodes = pa.Table.from_pylist([{
                "episode_index": 0, "length": 2,
                "videos/observation.images.top/chunk_index": 0,
                "videos/observation.images.top/file_index": 0,
                "stats/observation.state/mean": [0.0] * 9,
            }])
            pq.write_table(episodes, source / "meta/episodes/chunk-000/file-000.parquet")
            pq.write_table(pa.Table.from_pylist([{"task_index": 0, "task": "test"}]), source / "meta/tasks.parquet")
            video = source / "videos/observation.images.top/chunk-000/file-000.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"video-placeholder")
            phase = root / "phase.parquet"
            pq.write_table(pa.Table.from_pylist([
                {"episode_index": 0, "frame_index": 0, "phase_id": 0, "valid": True, "reviewed": True},
                {"episode_index": 0, "frame_index": 1, "phase_id": -1, "valid": False, "reviewed": True},
            ]), phase)

            manifest = prepare(source, destination, phase)
            output = pq.read_table(destination / "data/chunk-000/file-000.parquet")
            self.assertEqual(output["observation.state"].to_pylist(), [list(range(8)), list(range(1, 9))])
            self.assertTrue(output["action"].equals(data["action"]))
            self.assertEqual(json.loads((destination / "meta/info.json").read_text())["features"]["observation.state"]["shape"], [8])
            self.assertEqual(len(json.loads((destination / "meta/stats.json").read_text())["observation.state"]["mean"]), 8)
            self.assertTrue((destination / "videos/observation.images.top/chunk-000/file-000.mp4").samefile(video))
            self.assertTrue(all(manifest["checks"].values()))


if __name__ == "__main__":
    unittest.main()
