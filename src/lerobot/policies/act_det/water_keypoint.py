"""Sparse, visible-only water-reference supervision (pilot, not a cup mask)."""

import json
import math
from pathlib import Path

import torch


class WaterPointLabels:
    def __init__(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.width, self.height = data["image_size_wh"]
        self.labels = {}
        for row in data["frames"]:
            key = (row["episode"], row["frame"], row["camera"])
            if key in self.labels:
                raise ValueError(f"Duplicate water point: {key}")
            if type(row["visible"]) is not bool:
                raise ValueError("Annotated visibility must be boolean")
            point = row.get("point_xy")
            if row["visible"]:
                if point is None or len(point) != 2:
                    raise ValueError("Visible point requires x/y")
                x, y = point
                if not (math.isfinite(x) and math.isfinite(y)
                        and 0 <= x < self.width and 0 <= y < self.height):
                    raise ValueError("Water point outside original image")
            elif point is not None:
                raise ValueError("Occluded center must not have guessed coordinates")
            self.labels[key] = row

    def heatmaps(self, keys, resolution, device="cpu", sigma=2.0):
        """Gaussian targets; sigma in output-grid pixels. Missing != occluded."""
        if sigma <= 0:
            raise ValueError("Heatmap sigma must be positive")
        h, w = resolution
        target = torch.zeros((len(keys), 1, h, w), device=device)
        valid = torch.zeros(len(keys), dtype=torch.bool, device=device)
        yy, xx = torch.meshgrid(torch.arange(h, device=device),
                                torch.arange(w, device=device), indexing="ij")
        for index, key in enumerate(keys):
            row = self.labels.get(tuple(key))
            if row is None or not row["visible"]:
                continue
            x, y = row["point_xy"]
            # Pixel-center mapping consistent with align_corners=False resizing.
            x = (x + 0.5) * w / self.width - 0.5
            y = (y + 0.5) * h / self.height - 0.5
            target[index, 0] = torch.exp(-((xx-x)**2 + (yy-y)**2)/(2*sigma**2))
            valid[index] = True
        return target, valid


def water_keypoint_loss(logits, target, visible):
    """Heatmap MSE on visible frames only; no supervision of occluded outputs."""
    if logits.shape != target.shape or visible.shape != (logits.shape[0],):
        raise ValueError("Water heatmap/visibility shapes do not match")
    if visible.dtype != torch.bool:
        raise ValueError("Water visibility must be boolean")
    # Select before computing: even NaN placeholders in ignored frames cannot leak.
    prediction = logits[visible].float().sigmoid()
    if prediction.numel() == 0:
        return prediction.sum()
    return (prediction - target[visible].float()).square().mean()
