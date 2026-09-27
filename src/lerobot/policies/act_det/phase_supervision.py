"""Reviewed per-frame phase labels for ACTDet auxiliary supervision."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


class PhaseLabels:
    """Load immutable labels keyed by episode-local frame index."""

    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        table = pq.read_table(path, columns=["episode_index", "frame_index", "phase_id", "valid"])
        self.labels: dict[tuple[int, int], int] = {}
        for row in table.to_pylist():
            key = (int(row["episode_index"]), int(row["frame_index"]))
            if key in self.labels:
                raise ValueError(f"Duplicate phase label: {key}")
            phase_id = int(row["phase_id"])
            valid = bool(row["valid"])
            if valid != (0 <= phase_id < 4):
                raise ValueError(f"Invalid phase label at {key}: phase_id={phase_id}, valid={valid}")
            self.labels[key] = phase_id if valid else -1

    def targets(self, episode_index: Tensor, frame_index: Tensor, *, device: torch.device) -> Tensor:
        keys = zip(episode_index.detach().cpu().tolist(), frame_index.detach().cpu().tolist(), strict=True)
        return torch.tensor(
            [self.labels.get((int(episode), int(frame)), -1) for episode, frame in keys],
            dtype=torch.long,
            device=device,
        )


def phase_classification_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Cross entropy over reviewed frames; an all-unknown batch contributes zero."""

    valid = targets >= 0
    if not torch.any(valid):
        return logits.sum() * 0
    return F.cross_entropy(logits[valid], targets[valid])
