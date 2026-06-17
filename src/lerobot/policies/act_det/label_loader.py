"""CVAT XML annotation loader for detection labels.

Parses CVAT 1.1 XML annotation files and provides O(1) lookup of bounding boxes
by (episode_index, frame_index).

All annotations are loaded into memory on construction. The total memory footprint
is small (~2.6 MB for 180 episodes of 718 frames with 1 bbox/frame).

XML structure (CVAT 1.1):
    <annotations>
      <meta>
        <task>
          <labels>
            <label><name>cup</name></label>
          </labels>
        </task>
      </meta>
      <track id="0" label="cup">
        <box frame="0" xtl="100" ytl="200" xbr="300" ybr="400" ... />
        <box frame="1" ... />
      </track>
    </annotations>
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path


class LabelLoader:
    """Loads and caches CVAT XML detection annotations for fast per-frame lookup.

    Args:
        annotation_dir: Root directory containing per-camera annotation subdirs.
        camera_keys: Mapping from canonical camera key to subdirectory name.
                     e.g. {"observation.images.top": "top", "observation.images.wrist": "gripper"}
        enabled: Whether detection labels are available. If False, all lookups
                 return None.
    """

    def __init__(
        self,
        annotation_dir: str | None = None,
        camera_keys: dict[str, str] | None = None,
    ):
        self._cache: dict[int, dict[int, dict]] = {}  # episode_index -> frame_index -> labels
        self._enabled = annotation_dir is not None
        self._camera_keys = camera_keys or {}

        if not self._enabled:
            return

        self._annotation_dir = Path(annotation_dir)
        self._load_all()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def get_labels(
        self,
        camera_key: str,
        episode_index: int,
        frame_index: int,
    ) -> dict | None:
        """Get detection labels for a specific frame.

        Args:
            camera_key: Canonical camera key (e.g. "observation.images.top").
            episode_index: Episode index in the dataset.
            frame_index: Frame index within the episode.

        Returns:
            Dict with keys "labels" (list[str]) and "bboxes" (list[[x1,y1,x2,y2]]),
            or None if no annotations exist for this frame/camera.
        """
        if not self._enabled:
            return None

        camera_subdir = self._camera_keys.get(camera_key)
        if camera_subdir is None:
            return None

        key = self._make_key(camera_subdir, episode_index, frame_index)
        return self._cache.get(key)

    def _make_key(self, camera: str, episode: int, frame: int) -> tuple:
        return (camera, episode, frame)

    def _load_all(self):
        """Parse all XML files in the annotation directory and populate the cache."""
        if not self._annotation_dir.exists():
            return

        for camera_subdir in os.listdir(self._annotation_dir):
            camera_path = self._annotation_dir / camera_subdir
            if not camera_path.is_dir():
                continue

            # Map camera subdir name back to canonical key if possible.
            # Otherwise use the subdir name directly.
            camera_key = camera_subdir
            for canonical, subdir in self._camera_keys.items():
                if subdir == camera_subdir:
                    camera_key = canonical
                    break

            for filename in os.listdir(camera_path):
                if not filename.endswith(".xml"):
                    continue

                filepath = camera_path / filename
                try:
                    parsed = self._parse_xml(filepath)
                except Exception:
                    continue

                for entry in parsed:
                    key = self._make_key(camera_key, entry["episode_index"], entry["frame_index"])
                    self._cache[key] = {
                        "labels": entry["labels"],
                        "bboxes": entry["bboxes"],
                    }

    def _parse_xml(self, filepath: Path) -> list[dict]:
        """Parse a single CVAT XML file into per-frame annotation dicts.

        Extracts the episode index from the filename convention:
          episode_{index:03d}.xml

        Returns:
            List of dicts with keys: episode_index, frame_index, labels, bboxes.
        """
        filename = filepath.stem  # e.g. "episode_025"
        # Extract episode index from filename.
        parts = filename.split("_")
        try:
            episode_index = int(parts[-1])
        except (ValueError, IndexError):
            episode_index = 0

        tree = ET.parse(str(filepath))
        root = tree.getroot()

        # Collect all labeled tracks.
        labels_map: dict[str, str] = {}  # track_id -> label_name
        for track in root.findall("track"):
            track_id = track.get("id", "0")
            label_name = track.get("label", "unknown")
            labels_map[track_id] = label_name

        # Collect all box annotations per frame.
        frames: dict[int, list[tuple[str, float, float, float, float]]] = {}
        for track in root.findall("track"):
            track_id = track.get("id", "0")
            label = labels_map.get(track_id, track.get("label", "unknown"))
            for box in track.findall("box"):
                frame = int(box.get("frame", "0"))
                outside = box.get("outside", "0")
                if outside == "1":
                    continue  # Object is outside the frame.
                xtl = float(box.get("xtl", "0"))
                ytl = float(box.get("ytl", "0"))
                xbr = float(box.get("xbr", "0"))
                ybr = float(box.get("ybr", "0"))

                if frame not in frames:
                    frames[frame] = []
                frames[frame].append((label, xtl, ytl, xbr, ybr))

        # Build output.
        results = []
        for frame_idx, boxes in frames.items():
            labels_list = []
            bboxes_list = []
            for label, xtl, ytl, xbr, ybr in boxes:
                labels_list.append(label)
                bboxes_list.append([xtl, ytl, xbr, ybr])

            results.append({
                "episode_index": episode_index,
                "frame_index": frame_idx,
                "labels": labels_list,
                "bboxes": bboxes_list,
            })

        return results
