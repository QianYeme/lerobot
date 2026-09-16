"""Create timestamped contact sheets and action traces for manual phase review."""

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="数据集/formal1_C")
    parser.add_argument("--output", default="outputs/formal1_C50_phase1/master_gripper_ablation_100k/phase_review")
    args = parser.parse_args()
    root, output = Path(args.dataset_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fps = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))["fps"]
    episodes = [7, 11, 13, 14, 16, 27, 35, 38, 40, 43]
    metadata = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    table = pq.read_table(root / "data/chunk-000/file-000.parquet")
    actions = np.asarray(table["action"].to_pylist())
    episode_indices = np.asarray(table["episode_index"].to_pylist())
    frames = np.asarray(table["frame_index"].to_pylist())
    trace = {}
    phases = {"interval_convention": "episode-local [start, stop) frames", "episodes": {}}
    for episode in episodes:
        row = next(row for row in metadata if row["episode_index"] == episode)
        start = row["videos/observation.images.top/from_timestamp"]
        sample_frames = list(range(0, row["length"], 45))
        sheet = Image.new("RGB", (1280, ((len(sample_frames) + 3) // 4) * 264), "white")
        draw = ImageDraw.Draw(sheet)
        chunk = row["videos/observation.images.top/chunk_index"]
        file_index = row["videos/observation.images.top/file_index"]
        video = root / f"videos/observation.images.top/chunk-{chunk:03d}/file-{file_index:03d}.mp4"
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            container.seek(int(start / stream.time_base), stream=stream, backward=True)
            pending = 0
            for frame in container.decode(stream):
                if frame.time + 1e-4 < start + sample_frames[pending] / fps:
                    continue
                x, y = (pending % 4) * 320, (pending // 4) * 264
                sheet.paste(frame.to_image().resize((320, 240)), (x, y))
                f = sample_frames[pending]
                draw.text((x + 5, y + 242), f"ep {episode} frame {f} ({f / fps:.1f}s)", fill="black")
                pending += 1
                if pending == len(sample_frames):
                    break
            if pending != len(sample_frames):
                raise RuntimeError(f"Incomplete contact sheet for episode {episode}")
        sheet.save(output / f"episode_{episode:03d}.jpg", quality=90)
        selected = (episode_indices == episode) & (frames % 15 == 0)
        trace[str(episode)] = [
            {"frame": int(f), "action": np.round(a, 3).tolist()}
            for f, a in zip(frames[selected], actions[selected], strict=True)
        ]
        phases["episodes"][str(episode)] = {
            "length": int(row["length"]), "reviewed": False,
            "notes": "Replace unknown ranges only after checking video and action traces.",
            "segments": [{"start": 0, "stop": int(row["length"]), "phase": "unknown"}],
        }
        print(f"PHASE_REVIEW ep {episode}", flush=True)
    (output / "action_traces.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    phase_path = output / "phase_annotations.json"
    if not phase_path.exists():
        phase_path.write_text(json.dumps(phases, indent=2), encoding="utf-8")
    else:
        print(f"Preserved existing manual annotations: {phase_path}")


if __name__ == "__main__":
    main()
