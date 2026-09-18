"""Render neighboring frames around detected mask-area jumps for manual review."""

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from prepare_c50_mask_preview import boxes, extract_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[])
    args = parser.parse_args()
    fps = json.loads((args.dataset_root / "meta/info.json").read_text())["fps"]
    for record_path in sorted(args.preview.glob("episode_*.json")):
        record = json.loads(record_path.read_text())
        ep = record["episode"]
        flags = record["flags"]["area_jumps_over_50pct"]
        if not flags and not args.frames:
            continue
        indices = sorted({i for flag in flags for i in (flag - 1, flag, flag + 1) if 0 <= i < record["length"]}
                         | {i for i in args.frames if 0 <= i < record["length"]})
        labels, _ = boxes(args.dataset_root, ep)
        with np.load(args.preview / f"masks/top/episode_{ep:03d}.npz") as archive:
            masks = archive["masks"]
        sheet = Image.new("RGB", (1280, math.ceil(len(indices) / 2) * 510), "white")
        with tempfile.TemporaryDirectory(prefix="c50_flag_review_", dir=args.preview) as directory:
            for panel, index in enumerate(indices):
                extract_frames(Path(record["video"]), record["video_from_timestamp"] + index / fps, 1, fps, Path(directory))
                rgb = np.asarray(Image.open(Path(directory) / "000000.jpg")).copy()
                foreground = masks[index].astype(bool)
                rgb[foreground] = (0.65 * rgb[foreground] + 0.35 * np.array([0, 255, 0])).astype(np.uint8)
                image = Image.fromarray(rgb)
                ImageDraw.Draw(image).rectangle(labels[index], outline="red", width=2)
                x, y = panel % 2 * 640, panel // 2 * 510
                sheet.paste(image, (x, y + 30))
                ImageDraw.Draw(sheet).text((x + 8, y + 8), f"episode={ep} frame={index} pixels={int(masks[index].sum())}", fill="black")
        sheet.save(args.preview / f"episode_{ep:03d}_flagged_preview.jpg", quality=95)


if __name__ == "__main__":
    main()
