"""Create a Formal3 derivative pairing reset observations with first-motion actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from prepare_formal3_time_trim import feature_stats, replace_column


STATE = "observation.state"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def episode_table(table: pa.Table, episode: int) -> pa.Table:
    return table.filter(pc.equal(table["episode_index"], episode))


def remap_xml(source: Path, destination: Path, start: int, end: int, prefix: int) -> int:
    tree = ET.parse(source)
    root = tree.getroot()
    kept = 0
    for track in root.findall(".//track"):
        for box in list(track.findall("box")):
            frame = int(box.attrib["frame"])
            if 0 <= frame < prefix:
                new_frame = frame
            elif start + prefix <= frame < end:
                new_frame = frame - start
            else:
                track.remove(box)
                continue
            box.attrib["frame"] = str(new_frame)
            kept += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return kept


def encode_episode_video(job: dict) -> dict:
    source, destination = Path(job["source"]), Path(job["destination"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    fps = job["fps"]
    t0 = job["episode_offset"]
    prefix_end = t0 + job["prefix"] / fps
    continuation_start = t0 + (job["start"] + job["prefix"]) / fps
    continuation_end = t0 + job["end"] / fps
    graph = (
        f"[0:v]trim=start={t0:.9f}:end={prefix_end:.9f},setpts=PTS-STARTPTS[a];"
        f"[0:v]trim=start={continuation_start:.9f}:end={continuation_end:.9f},setpts=PTS-STARTPTS[b];"
        "[a][b]concat=n=2:v=1:a=0[v]"
    )
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-filter_complex", graph, "-map", "[v]", "-an", "-r", str(fps),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        str(destination),
    ]
    subprocess.run(command, check=True)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", str(destination)],
        check=True, capture_output=True, text=True,
    )
    frames = int(probe.stdout.strip())
    if frames != job["length"]:
        raise ValueError(f"Video frame count mismatch for {destination}: {frames} != {job['length']}")
    return {"path": str(destination), "frames": frames, "sha256": sha256(destination)}


def prepare(args: argparse.Namespace) -> dict:
    original = args.original.resolve()
    trimmed = args.trimmed.resolve()
    destination = args.destination.resolve()
    staging = destination.with_name(destination.name + ".preparing")
    if destination.exists() or staging.exists():
        raise FileExistsError(f"Refusing to overwrite {destination} or {staging}")

    original_data = pa.concat_tables([pq.read_table(path) for path in sorted((original / "data").glob("*/*.parquet"))])
    trimmed_data = pa.concat_tables([pq.read_table(path) for path in sorted((trimmed / "data").glob("*/*.parquet"))])
    original_episodes = pa.concat_tables([pq.read_table(path) for path in sorted((original / "meta/episodes").glob("*/*.parquet"))])
    trimmed_episodes = pa.concat_tables([pq.read_table(path) for path in sorted((trimmed / "meta/episodes").glob("*/*.parquet"))])
    info = json.loads((trimmed / "meta/info.json").read_text(encoding="utf-8"))
    trim_manifest = json.loads((trimmed / "meta/time_trim_manifest.json").read_text(encoding="utf-8"))
    boundaries = {row["episode_index"]: row for row in trim_manifest["boundaries"]}
    fps = int(info["fps"])
    cameras = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    if args.prefix_frames < 1:
        raise ValueError("prefix-frames must be positive")

    parts, episode_rows, video_jobs = [], [], []
    for episode in range(info["total_episodes"]):
        boundary = boundaries[episode]
        start, end, length = boundary["start"], boundary["end"], boundary["length"]
        if args.prefix_frames >= length:
            raise ValueError(f"Episode {episode} cannot support prefix {args.prefix_frames}")
        source_ep = episode_table(original_data, episode)
        target_ep = episode_table(trimmed_data, episode)
        reset_states = source_ep[STATE].slice(0, args.prefix_frames)
        continuation_states = source_ep[STATE].slice(start + args.prefix_frames, length - args.prefix_frames)
        paired_states = pa.concat_arrays([reset_states.combine_chunks(), continuation_states.combine_chunks()])
        target_ep = target_ep.set_column(target_ep.schema.get_field_index(STATE), STATE, paired_states)
        parts.append(target_ep)

        old_row = episode_table(trimmed_episodes, episode).to_pylist()[0]
        state_stats = feature_stats(np.asarray(paired_states.to_pylist()))
        for stat, value in state_stats.items():
            old_row[f"stats/{STATE}/{stat}"] = value
        source_meta = episode_table(original_episodes, episode).to_pylist()[0]
        for camera in cameras:
            prefix = f"videos/{camera}"
            old_row[f"{prefix}/chunk_index"] = 0
            old_row[f"{prefix}/file_index"] = episode
            old_row[f"{prefix}/from_timestamp"] = 0.0
            old_row[f"{prefix}/to_timestamp"] = length / fps
            source_relative = Path(json.loads((original / "meta/info.json").read_text())["video_path"].format(
                video_key=camera,
                chunk_index=source_meta[f"{prefix}/chunk_index"],
                file_index=source_meta[f"{prefix}/file_index"],
            ))
            destination_relative = Path(info["video_path"].format(video_key=camera, chunk_index=0, file_index=episode))
            video_jobs.append({
                "source": original / source_relative, "destination": staging / destination_relative,
                "episode_offset": float(source_meta[f"{prefix}/from_timestamp"]),
                "start": start, "end": end, "prefix": args.prefix_frames,
                "length": length, "fps": fps,
            })
        episode_rows.append(old_row)

    paired_data = pa.concat_tables(parts)
    paired_episodes = pa.Table.from_pylist(episode_rows, schema=trimmed_episodes.schema)
    info["features"][cameras[0]]["info"]["video.codec"] = "h264"
    for camera in cameras[1:]:
        info["features"][camera]["info"]["video.codec"] = "h264"

    (staging / "data/chunk-000").mkdir(parents=True)
    (staging / "meta/episodes/chunk-000").mkdir(parents=True)
    pq.write_table(paired_data.replace_schema_metadata(None), staging / "data/chunk-000/file-000.parquet")
    pq.write_table(paired_episodes.replace_schema_metadata(None), staging / "meta/episodes/chunk-000/file-000.parquet")
    (staging / "meta/info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(trimmed / "meta/tasks.parquet", staging / "meta/tasks.parquet")
    shutil.copy2(trimmed / "meta/reviewed_phase_labels.parquet", staging / "meta/reviewed_phase_labels.parquet")

    stats = json.loads((trimmed / "meta/stats.json").read_text(encoding="utf-8"))
    stats[STATE] = feature_stats(np.asarray(paired_data[STATE].to_pylist()))
    (staging / "meta/stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    annotation_boxes = 0
    for episode, boundary in boundaries.items():
        annotation_boxes += remap_xml(
            original / f"annotations/top/episode_{episode:03d}.xml",
            staging / f"annotations/top/episode_{episode:03d}.xml",
            boundary["start"], boundary["end"], args.prefix_frames,
        )

    with ThreadPoolExecutor(max_workers=args.video_workers) as pool:
        video_results = list(pool.map(encode_episode_video, video_jobs))

    checks = {
        "frames_preserved": len(paired_data) == len(trimmed_data),
        "actions_exact": paired_data["action"].equals(trimmed_data["action"]),
        "phase_rows_preserved": pq.read_table(staging / "meta/reviewed_phase_labels.parquet").num_rows == len(paired_data),
        "annotation_one_per_frame": annotation_boxes == len(paired_data),
        "videos_verified": len(video_results) == len(cameras) * info["total_episodes"],
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    manifest = {
        "schema_version": 1,
        "original": str(original), "trimmed": str(trimmed),
        "prefix_frames": args.prefix_frames, "prefix_seconds": args.prefix_frames / fps,
        "mapping": "observation[0:K]=original[0:K], action=original[S:E], observation[K:]=original[S+K:E]",
        "frames": len(paired_data), "episodes": info["total_episodes"],
        "video_encoding": "per-episode H.264 CRF18; exact decoded frame count verified",
        "video_sha256": {str(Path(item["path"]).relative_to(staging)): item["sha256"] for item in video_results},
        "checks": checks,
    }
    (staging / "meta/reset_pair_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    staging.rename(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--trimmed", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--prefix-frames", type=int, default=15)
    parser.add_argument("--video-workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
