"""NPZ mask loader for Mask-Guided Perception.

Loads SAM 2 pre-generated mask files (.npz) from the annotations directory
and provides O(1) lookup during training.

Mask files are expected in the following structure:
    {annotation_dir}/masks/{camera_subdir}/episode_{index:03d}.npz

Each .npz file contains a single array with shape (N_frames, H, W),
dtype float32, values in [0, 1] after SIGMOID + Gaussian blur.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

import numpy as np


class MaskLoader:
    """Loads and caches NPZ mask files for fast per-frame lookup.

    Uses an LRU cache to keep recently accessed episode masks in memory.
    Typical cache footprint: ~8 episodes × 718 frames × 480×640 × 4 bytes ≈ 3.5 GB.

    Args:
        mask_dir: Root directory containing per-camera mask NPZ files.
        camera_keys: Mapping from canonical camera key to subdirectory name.
        max_cache_episodes: Maximum number of episode mask arrays to keep in memory.
    """

    def __init__(
        self,
        mask_dir: str | None = None,
        camera_keys: dict[str, str] | None = None,
        max_cache_episodes: int = 8,
    ):
        self._cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._max_cache = max_cache_episodes
        self._camera_keys = camera_keys or {}
        self._enabled = mask_dir is not None and os.path.isdir(mask_dir)
        self._mask_dir = Path(mask_dir) if self._enabled else None

        # Build index: (camera, episode) → filepath.
        self._file_index: dict[tuple[str, int], Path] = {}
        if self._enabled:
            self._build_index()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def get_mask(
        self,
        camera_key: str,
        episode_index: int,
        frame_index: int,
    ) -> np.ndarray | None:
        """Get the SAM 2 mask for a specific frame.

        Args:
            camera_key: Canonical camera key (e.g. "observation.images.top").
            episode_index: Episode index in the dataset.
            frame_index: Frame index within the episode.

        Returns:
            (H, W) float32 numpy array with values in [0, 1], or None.
        """
        if not self._enabled:
            return None

        camera_subdir = self._camera_keys.get(camera_key)
        if camera_subdir is None:
            camera_subdir = camera_key.split(".")[-1]

        cache_key = (camera_subdir, episode_index)
        filepath = self._file_index.get(cache_key)
        if filepath is None:
            return None

        # Load entire episode mask array into cache if not present.
        if cache_key not in self._cache:
            try:
                data = np.load(str(filepath))
                # .npz files store arrays under 'arr_0' (default) or custom key.
                # Try common keys.
                if "arr_0" in data:
                    masks = data["arr_0"]
                elif "masks" in data:
                    masks = data["masks"]
                else:
                    # Take the first array.
                    masks = list(data.values())[0]

                masks = masks.astype(np.float32)

                # LRU eviction.
                if len(self._cache) >= self._max_cache:
                    self._cache.popitem(last=False)

                self._cache[cache_key] = masks
            except Exception:
                return None

        masks = self._cache[cache_key]

        # Move this entry to the end (most recently used).
        self._cache.move_to_end(cache_key)

        if frame_index < 0 or frame_index >= len(masks):
            return None

        return masks[frame_index]  # (H, W)

    def _build_index(self):
        """Scan mask directory and build (camera, episode) → filepath mapping."""
        for camera_subdir in os.listdir(self._mask_dir):
            camera_path = self._mask_dir / camera_subdir
            if not camera_path.is_dir():
                continue

            for filename in os.listdir(camera_path):
                if not filename.endswith(".npz"):
                    continue

                # Extract episode index from filename: episode_{index:03d}.npz
                stem = filename.replace(".npz", "")
                parts = stem.split("_")
                try:
                    episode_index = int(parts[-1])
                except (ValueError, IndexError):
                    continue

                key = (camera_subdir, episode_index)
                self._file_index[key] = camera_path / filename
