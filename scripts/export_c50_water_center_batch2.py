"""Export the second water-center marking batch (frame-205 counterexample + cross-episode frames).

Writes raw 640x480 PNGs for the user to mark with a single red dot at the visible
water-surface centre; occluded or uncertain frames should be reported as invisible
rather than guessed. Frame list follows C50_水面参考中心_初步核验_2026-09-18.md next steps.

Usage: export_c50_water_center_batch2.py [--root ...] [--output ...]
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

FRAMES = [(20, 204), (20, 205), (20, 206), (1, 350), (26, 350), (42, 350)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("数据集/formal1_C50_nomaster"))
    parser.add_argument("--repo-id", default="QYyyyyyyy/formal1_C50_nomaster")
    parser.add_argument("--output", type=Path, default=Path("outputs/water_center_batch2_20260918"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    for episode, frame in FRAMES:
        dataset = LeRobotDataset(args.repo_id, root=args.root, episodes=[episode], video_backend="pyav")
        item = dataset[frame]
        image = (item["observation.images.top"].numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
        path = args.output / f"episode_{episode:03d}_frame_{frame:06d}_raw.png"
        Image.fromarray(image).save(path)
        print(path, image.shape, flush=True)


if __name__ == "__main__":
    main()
