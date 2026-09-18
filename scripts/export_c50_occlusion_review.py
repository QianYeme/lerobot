"""Export training-only occlusion frames and candidate overlays for prompt review."""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from prepare_c50_mask_preview import boxes, extract_frames

FRAMES = {20: [240, 399, 400, 537, 538, 541, 543, 547], 49: [280, 314, 332, 340, 529]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--propagated", type=Path, required=True)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    split = json.loads((args.dataset_root / "meta/nomaster_manifest.json").read_text())
    if not set(FRAMES).issubset(split["train"]):
        raise ValueError("Review frames must be training-only")
    fps = json.loads((args.dataset_root / "meta/info.json").read_text())["fps"]
    args.output.mkdir(parents=True)
    entries = []
    for episode, indices in FRAMES.items():
        record = json.loads((args.independent / f"episode_{episode:03d}.json").read_text())
        labels, _ = boxes(args.dataset_root, episode)
        with np.load(args.propagated / f"masks/top/episode_{episode:03d}.npz") as archive:
            propagated = archive["masks"][indices]
        with np.load(args.independent / f"masks/top/episode_{episode:03d}.npz") as archive:
            independent = archive["masks"][indices]
        with tempfile.TemporaryDirectory(dir=args.output) as directory:
            for offset, frame in enumerate(indices):
                extract_frames(Path(record["video"]), record["video_from_timestamp"] + frame / fps,
                               1, fps, Path(directory))
                raw = Image.open(Path(directory) / "000000.jpg").convert("RGB")
                stem = f"episode_{episode:03d}_frame_{frame:06d}"
                raw.save(args.output / f"{stem}_raw.png")
                sheet = Image.new("RGB", (1920, 510), "white")
                for column, (title, mask) in enumerate(zip(
                        ["RAW + cup box", "VIDEO PROPAGATION", "INDEPENDENT FRAME"],
                        [None, propagated[offset], independent[offset]])):
                    pixels = np.asarray(raw).copy()
                    if mask is not None:
                        foreground = mask.astype(bool)
                        pixels[foreground] = (0.65*pixels[foreground] + 0.35*np.array([0,255,0])).astype(np.uint8)
                    image = Image.fromarray(pixels)
                    ImageDraw.Draw(image).rectangle(labels[frame], outline="red", width=2)
                    sheet.paste(image, (640*column, 30))
                    ImageDraw.Draw(sheet).text((640*column+8, 8), f"{stem}: {title}", fill="black")
                sheet.save(args.output / f"{stem}_comparison.jpg", quality=95)
                entries.append({"episode": episode, "frame": frame, "camera": "top",
                                "image": f"{stem}_raw.png", "cup_box": labels[frame],
                                "positive_points": [], "negative_points": [], "review_status": "pending"})
    report = {"coordinate_system": "Original 640x480 image; x right, y down; origin top-left",
              "label_rule": "Visible cup only; exclude gripper; do not hallucinate hidden cup pixels",
              "formal_training_ready": False, "frames": entries}
    (args.output / "prompts_to_review.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"exported_frames": len(entries), "output": str(args.output), "review_status": "pending"}))


if __name__ == "__main__":
    main()
