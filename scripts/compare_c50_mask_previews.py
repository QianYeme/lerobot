"""Compare label stability and box spill; these are QA proxies, not mask accuracy."""
import argparse
import json
from pathlib import Path

import numpy as np

from prepare_c50_mask_preview import boxes


def metrics(root, preview, episode):
    labels, _ = boxes(root, episode)
    record = json.loads((preview / f"episode_{episode:03d}.json").read_text())
    with np.load(preview / f"masks/top/episode_{episode:03d}.npz") as archive:
        masks = archive["masks"]
        valid = archive["valid"]
    spill = []
    for index, mask in enumerate(masks):
        x1, y1, x2, y2 = labels[index]
        left, top = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        right, bottom = min(640, int(np.ceil(x2)) + 1), min(480, int(np.ceil(y2)) + 1)
        area = int(mask.sum())
        spill.append((area - int(mask[top:bottom, left:right].sum())) / max(area, 1))
    return {"frames": len(masks), "valid_frames": int(valid.sum()), "flags": record["flags"],
            "mean_outside_box_fraction": float(np.mean(spill)),
            "p95_outside_box_fraction": float(np.percentile(spill, 95)),
            "frames_outside_box_over_10pct": int(np.sum(np.asarray(spill) > 0.1)),
            "selected_outside_box_fraction": {str(i): spill[i] for i in [120, 240, 339, 340, 341, 528, 529, 530, 717]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"warning": "Box spill and area jumps cannot establish segmentation accuracy or occlusion correctness",
              "episodes": {str(ep): {"baseline": metrics(args.dataset_root, args.baseline, ep),
                                     "candidate": metrics(args.dataset_root, args.candidate, ep)} for ep in [0, 20, 49]}}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
