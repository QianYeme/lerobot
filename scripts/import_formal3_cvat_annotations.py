"""Validate nested CVAT exports and install canonical top-camera cup boxes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pyarrow.parquet as pq


EPISODE_PATTERN = re.compile(r"(?:ep|_)(\d{3})\.zip$")
CAMERA_ALIASES = {"top": "top", "wrist": "gripper"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_exports(export_dir: Path) -> dict[str, dict[int, dict]]:
    records: dict[str, dict[int, dict]] = {"top": {}, "gripper": {}}
    for archive in sorted(export_dir.glob("*.zip")):
        with ZipFile(archive) as outer:
            for member in outer.namelist():
                if not member.endswith(".zip"):
                    continue
                match = EPISODE_PATTERN.search(member)
                camera = next((canonical for alias, canonical in CAMERA_ALIASES.items()
                               if f"/{alias}/" in f"/{member}"), None)
                if match is None or camera is None:
                    raise ValueError(f"Cannot map nested CVAT export: {archive.name}:{member}")
                episode = int(match.group(1))
                if episode in records[camera]:
                    raise ValueError(f"Duplicate {camera} episode {episode}")
                with ZipFile(BytesIO(outer.read(member))) as inner:
                    root = ET.fromstring(inner.read("annotations.xml"))
                task = root.find("./meta/task")
                if task is None or task.findtext("mode") != "interpolation":
                    raise ValueError(f"Expected interpolation task: {member}")
                tracks = root.findall("track")
                boxes = root.findall(".//box")
                records[camera][episode] = {
                    "archive": archive.name,
                    "member": member,
                    "task_name": task.findtext("name"),
                    "size": int(task.findtext("size")),
                    "tracks": tracks,
                    "boxes": boxes,
                }
    return records


def validate_record(record: dict, *, episode: int, length: int, width: int, height: int) -> dict:
    if record["size"] != length:
        raise ValueError(f"Episode {episode} CVAT size {record['size']} != dataset length {length}")
    labels = [track.get("label") for track in record["tracks"]]
    if labels not in ([], ["cup"]):
        raise ValueError(f"Episode {episode} has unexpected tracks: {labels}")

    active_by_frame: dict[int, ET.Element] = {}
    keyframes = 0
    outside = 0
    for box in record["boxes"]:
        frame = int(box.get("frame", "-1"))
        if not 0 <= frame < length:
            raise ValueError(f"Episode {episode} has out-of-range frame {frame}")
        if box.get("keyframe", "0") == "1":
            keyframes += 1
        if box.get("outside", "0") == "1":
            outside += 1
            continue
        coords = [float(box.get(name, "nan")) for name in ("xtl", "ytl", "xbr", "ybr")]
        if not all(math.isfinite(value) for value in coords):
            raise ValueError(f"Episode {episode} frame {frame} has non-finite coordinates")
        x1, y1, x2, y2 = coords
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Episode {episode} frame {frame} has invalid box {coords}")
        if frame in active_by_frame:
            raise ValueError(f"Episode {episode} frame {frame} has multiple active cup boxes")
        active_by_frame[frame] = box
    return {
        "active_by_frame": active_by_frame,
        "tracks": len(record["tracks"]),
        "active_frames": len(active_by_frame),
        "outside_boxes": outside,
        "keyframes": keyframes,
        "interpolated_active_frames": len(active_by_frame) - sum(
            box.get("keyframe", "0") == "1" for box in active_by_frame.values()
        ),
    }


def canonical_xml(active_by_frame: dict[int, ET.Element]) -> bytes:
    root = ET.Element("annotations")
    track = ET.SubElement(root, "track", {"id": "0", "label": "cup"})
    for frame, source in sorted(active_by_frame.items()):
        attributes = {
            "frame": str(frame),
            "outside": "0",
            "occluded": source.get("occluded", "0"),
            "keyframe": source.get("keyframe", "0"),
            "xtl": source.get("xtl"),
            "ytl": source.get("ytl"),
            "xbr": source.get("xbr"),
            "ybr": source.get("ybr"),
        }
        ET.SubElement(track, "box", attributes)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def install(export_dir: Path, dataset_root: Path) -> dict:
    export_dir = export_dir.resolve()
    dataset_root = dataset_root.resolve()
    destination = dataset_root / "annotations"
    staging = dataset_root / "annotations.preparing"
    if destination.exists() or staging.exists():
        raise FileExistsError(f"Refusing to overwrite {destination} or {staging}")

    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    episodes = {
        int(row["episode_index"]): int(row["length"])
        for row in pq.read_table(dataset_root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    }
    if set(episodes) != set(range(info["total_episodes"])):
        raise ValueError("Dataset episode indices are not contiguous")
    height, width, _ = info["features"]["observation.images.top"]["shape"]
    records = parse_exports(export_dir)
    expected = set(episodes)
    if set(records["top"]) != expected or set(records["gripper"]) != expected:
        raise ValueError("CVAT export does not contain exactly one top and gripper task per episode")

    camera_reports: dict[str, dict] = {}
    validated: dict[str, dict[int, dict]] = {"top": {}, "gripper": {}}
    for camera in ("top", "gripper"):
        incomplete = []
        totals = {"active_frames": 0, "keyframes": 0, "interpolated_active_frames": 0, "outside_boxes": 0}
        for episode, length in episodes.items():
            result = validate_record(
                records[camera][episode], episode=episode, length=length, width=width, height=height
            )
            validated[camera][episode] = result
            for key in totals:
                totals[key] += result[key]
            if result["active_frames"] != length:
                incomplete.append({"episode": episode, "active_frames": result["active_frames"], "length": length})
        camera_reports[camera] = {
            "episodes": len(records[camera]),
            **totals,
            "incomplete_episodes": incomplete,
        }

    if camera_reports["top"]["incomplete_episodes"]:
        raise ValueError("Top-camera annotations do not cover every dataset frame")

    top_dir = staging / "top"
    top_dir.mkdir(parents=True)
    annotation_hashes = {}
    for episode in sorted(episodes):
        output = top_dir / f"episode_{episode:03d}.xml"
        output.write_bytes(canonical_xml(validated["top"][episode]["active_by_frame"]))
        annotation_hashes[output.name] = sha256(output)

    manifest = {
        "schema_version": 1,
        "source_archives": {path.name: sha256(path) for path in sorted(export_dir.glob("*.zip"))},
        "dataset": str(dataset_root),
        "image_size": {"width": width, "height": height},
        "installed_camera": "observation.images.top",
        "installed_subdirectory": "top",
        "class": "cup",
        "annotation_mode": "CVAT interpolation; keyframes are human-set, other active boxes are interpolated",
        "top_gate": "PASS",
        "gripper_gate": "EXCLUDED_PENDING_REVIEW",
        "cameras": camera_reports,
        "annotation_sha256": annotation_hashes,
        "privacy": "Canonical XML excludes CVAT task metadata, usernames, emails, URLs, and attributes not used by training",
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    staging.rename(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(install(args.export_dir, args.dataset_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
