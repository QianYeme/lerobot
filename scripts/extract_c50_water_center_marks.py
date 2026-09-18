"""Extract user red-dot water-center marks from a marked folder (episode-agnostic).

Generalizes extract_c50_water_points.py (pilot, episode 20) to any set of
episode_frame files. Marker detection only; no automatic water recognition.
Frames the user reported as occluded are recorded with visible=false and null
coordinates instead of guessed points. QC: marked image must be the original
640x480 raw file plus red pixels only (no rescale, crop or re-encode).

Usage:
  extract_c50_water_center_marks.py --raw-dir outputs/water_center_batch2_20260918 \
    --marked-dir outputs/water_center_batch2_20260918_marked \
    --dataset 数据集/formal1_C50_nomaster --out outputs/water_center_batch2_20260918/annotations.json \
    --occluded 20:204 20:205 20:206 --occluded-note "water surface center occluded by gripper right jaw half"
"""
import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

NAME = re.compile(r"episode_(\d+)_frame_(\d+)")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def cup_boxes(dataset, episode):
    xml = dataset / f"annotations/top/episode_{episode:03d}.xml"
    result = {}
    for track in ET.parse(xml).getroot().findall("track"):
        if track.get("label") != "cup":
            continue
        for box in track.findall("box"):
            if box.get("outside", "0") != "0":
                continue
            frame = int(box.get("frame"))
            if frame in result:
                raise ValueError(f"Multiple cup boxes: episode {episode}, frame {frame}")
            result[frame] = [float(box.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")]
    return result, xml


def red_marker(image, path):
    pixels = np.asarray(image).astype(np.int16)
    red = (pixels[:, :, 0] > 140) & (pixels[:, :, 0] - pixels[:, :, 1] > 60) \
        & (pixels[:, :, 0] - pixels[:, :, 2] > 60)
    yy, xx = np.nonzero(red)
    if not 3 <= len(xx) <= 100 or np.ptp(xx) > 12 or np.ptp(yy) > 12:
        raise ValueError(f"Expected one small red marker in {path.name}")
    return red, float(xx.mean()), float(yy.mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--marked-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--occluded", nargs="*", default=[],
                        help="ep:frame pairs the user reported as occluded (no dot drawn)")
    parser.add_argument("--occluded-note", default="user reported occluded")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    occluded = {}
    for item in args.occluded:
        episode, frame = (int(part) for part in item.split(":"))
        occluded[(episode, frame)] = args.occluded_note

    entries, xml_hashes = [], {}
    for path in sorted(args.marked_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = NAME.match(path.name)
        if not match:
            continue
        episode, frame = int(match.group(1)), int(match.group(2))
        if (episode, frame) in occluded:
            raise ValueError(f"Frame reported both marked and occluded: {episode}:{frame}")
        raw_path = args.raw_dir / path.name
        if not raw_path.exists():
            raise FileNotFoundError(f"No raw counterpart for {path.name}")
        image = Image.open(path).convert("RGB")
        if image.size != (640, 480):
            raise ValueError(f"Original 640x480 coordinates required: {path.name}")
        raw = np.asarray(Image.open(raw_path).convert("RGB")).astype(np.int16)
        marked = np.asarray(image).astype(np.int16)
        changed = np.abs(raw - marked).sum(axis=2) > 10
        red, x, y = red_marker(image, path)
        if (changed & ~red).any():
            raise ValueError(f"Marked image differs from raw outside the red marker: {path.name}")
        boxes, xml = cup_boxes(args.dataset, episode)
        if frame not in boxes:
            raise ValueError(f"No cup box at {episode}:{frame}; available near: "
                             f"{[f for f in sorted(boxes) if abs(f - frame) < 40][:8]}")
        x1, y1, x2, y2 = boxes[frame]
        if not x1 <= x <= x2 or not y1 <= y <= y2:
            raise ValueError(f"Marker outside cup box at {episode}:{frame}")
        xml_hashes[str(episode)] = hashlib.sha256(xml.read_bytes()).hexdigest()
        entries.append({"episode": episode, "frame": frame, "camera": "top", "visible": True,
                        "point_xy": [x, y], "red_marker_pixels": int(red.sum()),
                        "cup_box": [x1, y1, x2, y2],
                        "box_relative_xy": [(x - x1) / (x2 - x1), (y - y1) / (y2 - y1)],
                        "offset_from_box_center_px": [x - (x1 + x2) / 2, y - (y1 + y2) / 2],
                        "marked_image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "source": "User red-dot label; extraction is marker detection, not water detection"})
    for (episode, frame), note in sorted(occluded.items()):
        boxes, xml = cup_boxes(args.dataset, episode)
        xml_hashes[str(episode)] = hashlib.sha256(xml.read_bytes()).hexdigest()
        entries.append({"episode": episode, "frame": frame, "camera": "top", "visible": False,
                        "point_xy": None, "note": note,
                        "source": "User-reported occlusion; no coordinates guessed"})

    report = {"target": "User-defined visible water reference center",
              "semantic_status": "Not independently established as physical water-surface center "
                                 "rather than cup-bottom projection",
              "coordinate_system": "Original 640x480 image, x right / y down",
              "training_only": True, "frames": sorted(entries, key=lambda e: (e["episode"], e["frame"])),
              "formal_training_ready": False, "annotation_xml_sha256": xml_hashes,
              "warning": "Sparse labels; do not treat as tracking stability, localization accuracy, "
                         "or grasp coordinates"}
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for entry in report["frames"]:
        if entry["visible"]:
            print(f"ep{entry['episode']:03d} f{entry['frame']:03d} visible point={entry['point_xy']} "
                  f"rel=({entry['box_relative_xy'][0]:.3f},{entry['box_relative_xy'][1]:.3f})")
        else:
            print(f"ep{entry['episode']:03d} f{entry['frame']:03d} occluded: {entry['note']}")


if __name__ == "__main__":
    main()
