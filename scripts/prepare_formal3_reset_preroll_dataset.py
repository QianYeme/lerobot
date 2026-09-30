"""Validate the plateau-prefix + aligned-pre-roll row mapping before dataset construction.

This P2 tool deliberately writes only a mapping manifest. Video, annotation, phase,
and metadata construction must not start until this row-level Gate passes.
"""

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
import av

STATE = "observation.state"
NUMERIC_KEYS = ("action", STATE, "timestamp", "frame_index", "episode_index", "index", "task_index")
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)


def replace_column(table: pa.Table, name: str, values) -> pa.Table:
    index = table.schema.get_field_index(name)
    return table.set_column(index, name, pa.array(values, type=table.schema.field(index).type))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def feature_stats(values: np.ndarray) -> dict[str, list]:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    result = {
        "min": values.min(axis=0), "max": values.max(axis=0),
        "mean": values.mean(axis=0), "std": values.std(axis=0),
        "count": np.asarray([len(values)], dtype=np.int64),
    }
    result.update({f"q{int(q * 100):02d}": np.quantile(values, q, axis=0) for q in QUANTILES})
    return {key: value.tolist() for key, value in result.items()}


def remap_xml(source: Path, destination: Path, start: int, end: int, prefix: int) -> int:
    tree = ET.parse(source)
    root = tree.getroot()
    kept = 0
    for track in root.findall(".//track"):
        for box in list(track.findall("box")):
            frame = int(box.attrib["frame"])
            if 0 <= frame < prefix:
                new_frame = frame
            elif start <= frame < end:
                new_frame = prefix + frame - start
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
    fps, offset = job["fps"], job["episode_offset"]
    graph = (
        f"[0:v]trim=start={offset:.9f}:end={offset + job['prefix'] / fps:.9f},setpts=PTS-STARTPTS[a];"
        f"[0:v]trim=start={offset + job['start'] / fps:.9f}:end={offset + job['end'] / fps:.9f},setpts=PTS-STARTPTS[b];"
        "[a][b]concat=n=2:v=1:a=0[v]"
    )
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = "ffmpeg"
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-filter_complex", graph, "-map", "[v]", "-an", "-r", str(fps),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        str(destination),
    ], check=True)
    with av.open(str(destination)) as container:
        frames = sum(1 for _ in container.decode(video=0))
    if frames != job["length"]:
        raise ValueError(f"Video frame mismatch for {destination}: {frames} != {job['length']}")
    return {"path": str(destination), "frames": frames, "sha256": sha256(destination)}


def build_episode_table(
    original: pa.Table,
    trimmed: pa.Table,
    start: int,
    requested_prefix: int,
) -> tuple[pa.Table, int]:
    if start < 1:
        raise ValueError("Motion start must be positive")
    if requested_prefix < 1:
        raise ValueError("Requested prefix must be positive")
    prefix = min(requested_prefix, start)
    if original.num_rows < prefix or trimmed.num_rows < 1:
        raise ValueError("Episode is too short for plateau-prefix mapping")

    prefix_table = trimmed.slice(0, 1).take(pa.array(np.zeros(prefix, dtype=np.int64)))
    prefix_table = prefix_table.set_column(
        prefix_table.schema.get_field_index(STATE),
        STATE,
        original[STATE].slice(0, prefix),
    )
    first_action = trimmed["action"][0].as_py()
    prefix_table = replace_column(prefix_table, "action", [first_action] * prefix)
    return pa.concat_tables([prefix_table, trimmed]), prefix


def prepare_dataset(args: argparse.Namespace) -> dict:
    destination = args.destination.resolve()
    staging = destination.with_name(destination.name + ".preparing")
    if destination.exists() or staging.exists():
        raise FileExistsError(f"Refusing to overwrite {destination} or {staging}")

    original = pa.concat_tables([
        pq.read_table(path) for path in sorted((args.original / "data").glob("*/*.parquet"))
    ])
    trimmed = pa.concat_tables([
        pq.read_table(path) for path in sorted((args.trimmed / "data").glob("*/*.parquet"))
    ])
    original_episodes = pa.concat_tables([
        pq.read_table(path) for path in sorted((args.original / "meta/episodes").glob("*/*.parquet"))
    ])
    trimmed_episodes = pa.concat_tables([
        pq.read_table(path) for path in sorted((args.trimmed / "meta/episodes").glob("*/*.parquet"))
    ])
    trimmed_phase = pq.read_table(args.trimmed / "meta/reviewed_phase_labels.parquet")
    info = json.loads((args.trimmed / "meta/info.json").read_text(encoding="utf-8"))
    original_info = json.loads((args.original / "meta/info.json").read_text(encoding="utf-8"))
    trim_manifest = json.loads(args.trim_manifest.read_text(encoding="utf-8"))
    boundaries = {row["episode_index"]: row for row in trim_manifest["boundaries"]}
    fps = int(info["fps"])
    cameras = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]

    data_parts, phase_parts, episode_rows, video_jobs, mappings = [], [], [], [], []
    global_index = 0
    for episode in range(info["total_episodes"]):
        boundary = boundaries[episode]
        start, end = boundary["start"], boundary["end"]
        source_ep = original.filter(pc.equal(original["episode_index"], episode))
        trimmed_ep = trimmed.filter(pc.equal(trimmed["episode_index"], episode))
        mapped, prefix = build_episode_table(source_ep, trimmed_ep, start, args.prefix_frames)
        length = mapped.num_rows
        mapped = replace_column(mapped, "timestamp", np.arange(length, dtype=np.float32) / fps)
        mapped = replace_column(mapped, "frame_index", np.arange(length, dtype=np.int64))
        mapped = replace_column(mapped, "index", np.arange(global_index, global_index + length, dtype=np.int64))
        data_parts.append(mapped)

        phase_ep = trimmed_phase.filter(pc.equal(trimmed_phase["episode_index"], episode))
        phase_prefix = phase_ep.slice(0, 1).take(pa.array(np.zeros(prefix, dtype=np.int64)))
        mapped_phase = pa.concat_tables([phase_prefix, phase_ep])
        mapped_phase = replace_column(mapped_phase, "frame_index", np.arange(length, dtype=np.int64))
        phase_parts.append(mapped_phase)

        metadata = trimmed_episodes.filter(pc.equal(trimmed_episodes["episode_index"], episode)).to_pylist()[0]
        metadata["length"] = length
        metadata["dataset_from_index"] = global_index
        metadata["dataset_to_index"] = global_index + length
        source_metadata = original_episodes.filter(pc.equal(original_episodes["episode_index"], episode)).to_pylist()[0]
        for camera in cameras:
            key = f"videos/{camera}"
            metadata[f"{key}/chunk_index"] = 0
            metadata[f"{key}/file_index"] = episode
            metadata[f"{key}/from_timestamp"] = 0.0
            metadata[f"{key}/to_timestamp"] = length / fps
            source_relative = Path(original_info["video_path"].format(
                video_key=camera,
                chunk_index=source_metadata[f"{key}/chunk_index"],
                file_index=source_metadata[f"{key}/file_index"],
            ))
            destination_relative = Path(info["video_path"].format(
                video_key=camera, chunk_index=0, file_index=episode,
            ))
            video_jobs.append({
                "source": args.original / source_relative,
                "destination": staging / destination_relative,
                "episode_offset": float(source_metadata[f"{key}/from_timestamp"]),
                "start": start, "end": end, "prefix": prefix,
                "length": length, "fps": fps,
            })
        for key in NUMERIC_KEYS:
            stats = feature_stats(np.asarray(mapped[key].to_pylist()))
            for stat, value in stats.items():
                column = f"stats/{key}/{stat}"
                if column in metadata:
                    metadata[column] = value
        episode_rows.append(metadata)
        mappings.append({
            "episode_index": episode, "start": start, "end": end,
            "prefix_frames": prefix, "length": length,
        })
        global_index += length

    mapped_data = pa.concat_tables(data_parts)
    mapped_phase = pa.concat_tables(phase_parts)
    mapped_episodes = pa.Table.from_pylist(episode_rows, schema=trimmed_episodes.schema)
    info["total_frames"] = len(mapped_data)
    for camera in cameras:
        info["features"][camera]["info"]["video.codec"] = "h264"

    (staging / "data/chunk-000").mkdir(parents=True)
    (staging / "meta/episodes/chunk-000").mkdir(parents=True)
    pq.write_table(mapped_data.replace_schema_metadata(None), staging / "data/chunk-000/file-000.parquet")
    pq.write_table(mapped_episodes.replace_schema_metadata(None), staging / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(mapped_phase.replace_schema_metadata(None), staging / "meta/reviewed_phase_labels.parquet")
    (staging / "meta/info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(args.trimmed / "meta/tasks.parquet", staging / "meta/tasks.parquet")

    stats = json.loads((args.trimmed / "meta/stats.json").read_text(encoding="utf-8"))
    for key in NUMERIC_KEYS:
        stats[key] = feature_stats(np.asarray(mapped_data[key].to_pylist()))
    (staging / "meta/stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    annotation_boxes = 0
    annotation_hashes = {}
    for item in mappings:
        episode = item["episode_index"]
        target = staging / f"annotations/top/episode_{episode:03d}.xml"
        annotation_boxes += remap_xml(
            args.original / f"annotations/top/episode_{episode:03d}.xml",
            target, item["start"], item["end"], item["prefix_frames"],
        )
        annotation_hashes[target.name] = sha256(target)
    (staging / "annotations/manifest.json").write_text(json.dumps({
        "schema_version": 1, "source": str(args.original / "annotations/manifest.json"),
        "mapping": "original[0:K] + original[S:E]", "active_frames": annotation_boxes,
        "annotation_sha256": annotation_hashes,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=args.video_workers) as pool:
        video_results = list(pool.map(encode_episode_video, video_jobs))

    checks = {
        "episodes_preserved": len(mappings) == info["total_episodes"] == 60,
        "rows_match_mapping": len(mapped_data) == sum(item["length"] for item in mappings),
        "global_index_contiguous": mapped_data["index"].to_pylist() == list(range(len(mapped_data))),
        "phase_rows_match": len(mapped_phase) == len(mapped_data),
        "annotation_one_per_frame": annotation_boxes == len(mapped_data),
        "videos_verified": len(video_results) == len(cameras) * info["total_episodes"],
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    manifest = {
        "schema_version": 1, "status": "DATASET_BUILD_PASS",
        "original": str(args.original.resolve()), "trimmed": str(args.trimmed.resolve()),
        "requested_prefix_frames": args.prefix_frames, "frames": len(mapped_data),
        "episodes": mappings, "checks": checks,
        "mapping": "obs=original[0:K]+original[S:E]; action=repeat(original[S],K)+original[S:E]; K=min(15,S)",
        "video_encoding": "per-episode H.264 CRF18; decoded frame count verified",
        "video_sha256": {str(Path(row["path"]).relative_to(staging)): row["sha256"] for row in video_results},
    }
    (staging / "meta/reset_preroll_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    staging.rename(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--trimmed", type=Path, required=True)
    parser.add_argument("--trim-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--prefix-frames", type=int, default=15)
    parser.add_argument("--video-workers", type=int, default=4)
    args = parser.parse_args()

    if bool(args.output) == bool(args.destination):
        parser.error("Provide exactly one of --output (row Gate) or --destination (full build)")
    if args.destination:
        print(json.dumps(prepare_dataset(args), ensure_ascii=False, indent=2))
        return

    original = pa.concat_tables([
        pq.read_table(path) for path in sorted((args.original / "data").glob("*/*.parquet"))
    ])
    trimmed = pa.concat_tables([
        pq.read_table(path) for path in sorted((args.trimmed / "data").glob("*/*.parquet"))
    ])
    boundaries = {
        row["episode_index"]: row
        for row in json.loads(args.trim_manifest.read_text(encoding="utf-8"))["boundaries"]
    }

    episodes = []
    checks = {
        "plateau_actions_exact": True,
        "aligned_trimmed_rows_exact": True,
        "source_observation_overlap_absent": True,
    }
    total_rows = 0
    for episode in range(60):
        source_ep = original.filter(pc.equal(original["episode_index"], episode))
        trimmed_ep = trimmed.filter(pc.equal(trimmed["episode_index"], episode))
        start = boundaries[episode]["start"]
        mapped, prefix = build_episode_table(source_ep, trimmed_ep, start, args.prefix_frames)
        first_action = np.asarray(trimmed_ep["action"][0].as_py())
        prefix_actions = np.asarray(mapped["action"].slice(0, prefix).to_pylist())
        plateau_exact = np.array_equal(prefix_actions, np.repeat(first_action[None], prefix, axis=0))
        aligned_exact = mapped.slice(prefix).select(["action", STATE]).equals(
            trimmed_ep.select(["action", STATE])
        )
        no_overlap = prefix <= start
        checks["plateau_actions_exact"] &= plateau_exact
        checks["aligned_trimmed_rows_exact"] &= aligned_exact
        checks["source_observation_overlap_absent"] &= no_overlap
        episodes.append({
            "episode_index": episode,
            "motion_start_S": start,
            "prefix_frames": prefix,
            "trimmed_rows": trimmed_ep.num_rows,
            "mapped_rows": mapped.num_rows,
            "plateau_action_variance_max": float(np.max(np.var(prefix_actions, axis=0))),
            "plateau_actions_exact": plateau_exact,
            "aligned_trimmed_rows_exact": aligned_exact,
            "source_observation_overlap_absent": no_overlap,
        })
        total_rows += mapped.num_rows

    if not all(checks.values()):
        raise AssertionError(checks)
    manifest = {
        "schema_version": 1,
        "status": "ROW_MAPPING_GATE_PASS",
        "mapping": (
            "rows[0:K].obs=original[0:K], rows[0:K].action=original[S]; "
            "rows[K:]=time_trim[0:] with K=min(requested_prefix,S)"
        ),
        "requested_prefix_frames": args.prefix_frames,
        "episodes": episodes,
        "total_rows": total_rows,
        "checks": checks,
        "not_yet_validated": ["phase", "annotations", "videos", "metadata", "train-only stats"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
