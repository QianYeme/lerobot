#!/usr/bin/env python3
"""Generate SAM 2 segmentation masks for Mask-Guided Perception.

For each episode in a LeRobot dataset:
  1. Read per-episode metadata (frame count, data offset) from the episodes parquet.
  2. Extract the first-frame bbox from the CVAT XML annotation.
  3. Seek to the episode's frame range in the combined MP4 video and load only
     those frames.
  4. Write episode frames to a temporary MP4 file.
  5. Run SAM 2 video prediction with the first-frame box prompt.
  6. Apply Gaussian blur to mask edges.
  7. Save as compressed NPZ: annotations/masks/{camera}/episode_{index:03d}.npz

Usage:
    pip install opencv-python scipy
    pip install --no-build-isolation git+https://github.com/facebookresearch/sam2.git

    # Download model weights first:
    #   sam2.1_hiera_large.pt  →  checkpoints/
    #   (from https://github.com/facebookresearch/sam2)

    python scripts/generate_sam2_masks.py \\
        --dataset_root /path/to/dataset \\
        --cameras top gripper \\
        --gaussian_sigma 2.0

    # Dry run:
    python scripts/generate_sam2_masks.py \\
        --dataset_root /path/to/dataset \\
        --dry_run

Requirements:
    - torch >= 2.5.1 (already in lerobot conda env)
    - opencv-python, scipy, numpy (pip install)
    - segment-anything-2 from GitHub
    - SAM 2.1 model checkpoint (~850 MB)
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SAM 2 masks for all episodes in a LeRobot dataset."
    )
    parser.add_argument(
        "--dataset_root", type=str, required=True,
        help="Root directory of the LeRobot dataset (contains videos/ and annotations/)."
    )
    parser.add_argument(
        "--annotation_dir", type=str, default="annotations",
        help="Subdirectory with CVAT XML files, relative to dataset_root."
    )
    parser.add_argument(
        "--output_dir", type=str, default="annotations/masks",
        help="Output directory for NPZ mask files, relative to dataset_root."
    )
    parser.add_argument(
        "--cameras", nargs="+", default=["top", "gripper"],
        help="Camera subdirectory names under annotations/.  Default: top gripper."
    )
    parser.add_argument(
        "--gaussian_sigma", type=float, default=2.0,
        help="Sigma for Gaussian blur on mask edges.  Default: 2.0."
    )
    parser.add_argument(
        "--sam2_checkpoint", type=str,
        default="checkpoints/sam2.1_hiera_large.pt",
        help="Path to SAM 2.1 model checkpoint."
    )
    parser.add_argument(
        "--sam2_config", type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM 2.1 model config name (resolved inside the sam2 package)."
    )
    parser.add_argument(
        "--episodes", type=int, nargs="+", default=None,
        help="Specific episode indices to process (default: all found)."
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be done without running SAM 2."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip episodes that already have output NPZ files."
    )
    parser.add_argument(
        "--temp_dir", type=str, default=None,
        help="Directory for temporary per-episode video files."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
#  Annotation helpers
# ---------------------------------------------------------------------------

def extract_first_bbox(xml_path: Path) -> tuple[float, float, float, float] | None:
    """Extract the first frame's bounding box from a CVAT 1.1 XML file.

    Returns:
        ``(x1, y1, x2, y2)`` in pixel coordinates, or *None*.
    """
    if not xml_path.exists():
        return None

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    for track in root.findall("track"):
        boxes = track.findall("box")
        if not boxes:
            continue

        # Prefer frame 0; otherwise earliest frame.
        box0 = None
        for box in boxes:
            if box.get("frame") == "0":
                box0 = box
                break
        if box0 is None:
            box0 = boxes[0]

        xtl = float(box0.get("xtl", "0"))
        ytl = float(box0.get("ytl", "0"))
        xbr = float(box0.get("xbr", "0"))
        ybr = float(box0.get("ybr", "0"))
        return (xtl, ytl, xbr, ybr)

    return None


# ---------------------------------------------------------------------------
#  Episode metadata (from LeRobot parquet)
# ---------------------------------------------------------------------------

def load_episode_metadata(dataset_root: Path) -> dict[int, dict]:
    """Read per-episode frame counts and data offsets from the episodes parquet.

    Returns:
        Dict mapping ``episode_index`` → ``{"length": int, "data_from": int}``.
        Empty dict if the parquet cannot be read.
    """
    ep_parquet = dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not ep_parquet.exists():
        print(f"  [WARN] Episodes parquet not found: {ep_parquet}")
        return {}

    try:
        import pyarrow.parquet as pq
        table = pq.read_table(ep_parquet, columns=["episode_index", "length",
                                                     "dataset_from_index"])
        df = table.to_pandas()
        meta: dict[int, dict] = {}
        for _, row in df.iterrows():
            ep = int(row["episode_index"])
            meta[ep] = {
                "length": int(row["length"]),
                "data_from": int(row["dataset_from_index"]),
            }
        return meta
    except Exception as exc:
        print(f"  [WARN] Cannot read episodes parquet: {exc}")
        return {}


# ---------------------------------------------------------------------------
#  Video helpers
# ---------------------------------------------------------------------------

def _get_video_dims(video_path: Path) -> tuple[int, int] | None:
    """Return (width, height) of the first video stream, or None."""
    import json
    import subprocess

    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "json",
             str(video_path)],
            capture_output=True, text=True, timeout=15,
        )
        info = json.loads(result.stdout)
        stream = info["streams"][0]
        return int(stream["width"]), int(stream["height"])
    except Exception:
        return None


def load_episode_frames(
    video_path: Path,
    start_frame: int,
    num_frames: int,
    fps: int = 30,
) -> list[np.ndarray] | None:
    """Extract a contiguous range of frames from an MP4 using ffmpeg.

    Uses ffmpeg for frame seeking / decoding because it ships a software
    AV1 decoder, unlike the system OpenCV build.

    Args:
        video_path: Path to the combined MP4 file.
        start_frame: 0-indexed frame to start reading from.
        num_frames: Maximum number of frames to read.
        fps: Video frame rate (default 30).

    Returns:
        List of ``(H, W, 3)`` uint8 RGB frames, or *None* on failure.
    """
    import subprocess

    dims = _get_video_dims(video_path)
    if dims is None:
        return None
    W, H = dims

    start_sec = start_frame / fps
    duration_sec = (num_frames + 1) / fps  # +1 to ensure we don't drop the last frame

    cmd = [
        "ffmpeg",
        "-ss", f"{start_sec:.6f}",
        "-i", str(video_path),
        "-t", f"{duration_sec:.6f}",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-vcodec", "rawvideo",
        "-an", "-sn",
        "-nostdin", "-y",
        "-",
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        raw = proc.stdout.read()
        proc.wait(timeout=30)
    except Exception:
        return None

    if not raw:
        return None

    # Each frame: H * W * 3 bytes (rgb24).
    frame_bytes = H * W * 3
    total_frames = len(raw) // frame_bytes

    frames: list[np.ndarray] = []
    for i in range(min(total_frames, num_frames)):
        start = i * frame_bytes
        end = start + frame_bytes
        frame = np.frombuffer(raw[start:end], dtype=np.uint8).reshape(H, W, 3).copy()
        frames.append(frame)

    return frames if frames else None


def write_episode_video(
    frames: list[np.ndarray],
    output_path: Path,
    fps: int = 30,
) -> bool:
    """Write a list of RGB frames to an MP4 file using ffmpeg.

    Returns:
        *True* on success.
    """
    import subprocess

    if not frames:
        return False

    H, W = frames[0].shape[:2]

    # Pipe raw rgb24 frames into ffmpeg → encode to H.264 MP4.
    cmd = [
        "ffmpeg",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        "-crf", "23",
        "-y",
        "-nostdin",
        str(output_path),
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for frame in frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait(timeout=120)
        return proc.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
#  SAM 2 inference
# ---------------------------------------------------------------------------

def generate_masks_sam2(
    episode_video_path: Path,
    bbox: tuple[float, float, float, float],
    sam2_predictor,
    gaussian_sigma: float = 2.0,
) -> np.ndarray | None:
    """Run SAM 2 video prediction with a first-frame box prompt.

    Args:
        episode_video_path: Path to a temporary MP4 containing only the
            target episode's frames.
        bbox: ``(x1, y1, x2, y2)`` prompt for frame 0.
        sam2_predictor: SAM 2 video predictor instance.
        gaussian_sigma: Sigma for Gaussian blur on mask edges.

    Returns:
        ``(N_frames, H, W)`` float32 mask array, values in [0, 1], or *None*.
    """
    from scipy.ndimage import gaussian_filter
    import torch

    try:
        with torch.inference_mode():
            inference_state = sam2_predictor.init_state(
                video_path=str(episode_video_path),
            )

            _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                box=list(bbox),
            )

            # Propagate through all frames.
            masks: dict[int, np.ndarray] = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in \
                    sam2_predictor.propagate_in_video(inference_state):
                mask = (out_mask_logits[0][0] > 0.0).cpu().numpy().astype(np.float32)
                if gaussian_sigma > 0:
                    mask = gaussian_filter(mask, sigma=gaussian_sigma)
                masks[out_frame_idx] = mask

            if not masks:
                return None

            # Stack in frame order.
            num_frames = max(masks.keys()) + 1
            H, W = next(iter(masks.values())).shape
            stacked = np.zeros((num_frames, H, W), dtype=np.float32)
            for idx, m in masks.items():
                stacked[idx] = m

            return stacked

    except Exception as exc:
        print(f"  [ERROR] SAM 2 prediction failed: {exc}")
        return None


# ---------------------------------------------------------------------------
#  Episode discovery
# ---------------------------------------------------------------------------

def find_episodes(annotation_dir: Path, cameras: list[str]) -> set[int]:
    """Find all episode indices that have CVAT XML annotation files."""
    episodes: set[int] = set()
    for camera in cameras:
        cam_dir = annotation_dir / camera
        if not cam_dir.is_dir():
            continue
        for fname in os.listdir(cam_dir):
            if not fname.endswith(".xml"):
                continue
            stem = fname.replace(".xml", "")
            parts = stem.split("_")
            try:
                episodes.add(int(parts[-1]))
            except (ValueError, IndexError):
                pass
    return episodes


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    annotation_dir = dataset_root / args.annotation_dir
    output_dir = dataset_root / args.output_dir
    video_dir = dataset_root / "videos"

    # ---- Episode metadata (frame counts) ----
    ep_meta = load_episode_metadata(dataset_root)

    # ---- Discover episodes ----
    episodes = find_episodes(annotation_dir, args.cameras)
    if args.episodes:
        episodes = episodes & set(args.episodes)

    if not episodes:
        print("No episodes found with annotations.")
        print(f"  annotation_dir: {annotation_dir}")
        print(f"  cameras: {args.cameras}")
        print("  Expected: {annotation_dir}/{camera}/episode_XXX.xml")
        return

    print(f"Dataset: {dataset_root}")
    print(f"Episodes with annotations: {len(episodes)}")
    if ep_meta:
        lengths = [ep_meta[ep]["length"] for ep in sorted(episodes) if ep in ep_meta]
        if lengths:
            print(f"Frames per episode: min={min(lengths)}, max={max(lengths)}, "
                  f"mean={sum(lengths)/len(lengths):.0f}")
    print()

    # ---- Build per-episode plan ----
    plan: list[dict] = []  # each: {ep, camera, bbox, start_frame, num_frames, out_file}
    for ep in sorted(episodes):
        meta = ep_meta.get(ep, {})
        ep_length = meta.get("length")
        ep_data_from = meta.get("data_from", 0)

        for camera in args.cameras:
            xml_path = annotation_dir / camera / f"episode_{ep:03d}.xml"
            bbox = extract_first_bbox(xml_path)
            out_file = output_dir / camera / f"episode_{ep:03d}.npz"
            video_path = video_dir / f"observation.images.{camera}" / \
                         "chunk-000" / "file-000.mp4"

            plan.append({
                "ep": ep,
                "camera": camera,
                "bbox": bbox,
                "xml_path": xml_path,
                "video_path": video_path,
                "out_file": out_file,
                "start_frame": ep_data_from,
                "num_frames": ep_length,
            })

    # ---- Dry run ----
    if args.dry_run:
        print("[Dry run] Would process:")
        ok = skip = fail = 0
        for item in plan:
            if item["bbox"] is None:
                print(f"  Episode {item['ep']:03d} {item['camera']}: "
                      f"SKIP (no bbox in {item['xml_path']})")
                skip += 1
            elif args.resume and item["out_file"].exists():
                print(f"  Episode {item['ep']:03d} {item['camera']}: "
                      f"SKIP (exists: {item['out_file']})")
                skip += 1
            elif not item["video_path"].exists():
                print(f"  Episode {item['ep']:03d} {item['camera']}: "
                      f"FAIL (video missing: {item['video_path']})")
                fail += 1
            elif item["num_frames"] is None:
                print(f"  Episode {item['ep']:03d} {item['camera']}: "
                      f"FAIL (unknown frame count — parquet missing?)")
                fail += 1
            else:
                print(f"  Episode {item['ep']:03d} {item['camera']}: "
                      f"OK  bbox={item['bbox']}, "
                      f"frames=[{item['start_frame']},{item['start_frame']+item['num_frames']}), "
                      f"count={item['num_frames']}")
                ok += 1
        print(f"\n  Summary: OK={ok}  Skip={skip}  Fail={fail}")
        return

    # ---- Init SAM 2 ----
    print("Initializing SAM 2...")
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    checkpoint_path = args.sam2_checkpoint
    if not os.path.exists(checkpoint_path):
        # Try sam2 package's built-in location
        import sam2 as _sam2
        alt = os.path.join(os.path.dirname(_sam2.__file__), checkpoint_path)
        if os.path.exists(alt):
            checkpoint_path = alt

    if not os.path.exists(checkpoint_path):
        print(f"\n[ERROR] SAM 2 checkpoint not found: {checkpoint_path}")
        print("Download from: "
              "https://github.com/facebookresearch/sam2?tab=readme-ov-file"
              "#download-checkpoints")
        print(f"\nExpected at: {os.path.abspath(args.sam2_checkpoint)}")
        sys.exit(1)

    sam2_predictor = build_sam2_video_predictor(
        args.sam2_config,
        checkpoint_path,
        device=device,
    )
    print("  SAM 2 loaded.\n")

    # ---- Process ----
    total_ok, total_skip, total_fail = 0, 0, 0

    for item in plan:
        ep = item["ep"]
        camera = item["camera"]

        # Validate.
        if item["bbox"] is None:
            print(f"Episode {ep:03d} {camera}: SKIP (no bbox)")
            total_skip += 1
            continue
        # Clean up leftover tmp files from previous interrupted runs.
        tmp_file = item["out_file"].with_suffix(".tmp.npz")
        if tmp_file.exists():
            tmp_file.unlink()
        # Also clean old-format .npz.tmp and .npz.tmp.npz leftovers from
        # the early bug (numpy auto-appends .npz).
        for old_suffix in (".npz.tmp", ".npz.tmp.npz"):
            old_tmp = item["out_file"].with_suffix(old_suffix)
            if old_tmp.exists():
                old_tmp.unlink()

        if args.resume and item["out_file"].exists():
            print(f"Episode {ep:03d} {camera}: SKIP (exists)")
            total_skip += 1
            continue
        if not item["video_path"].exists():
            print(f"Episode {ep:03d} {camera}: FAIL (video missing: "
                  f"{item['video_path']})")
            total_fail += 1
            continue
        if item["num_frames"] is None:
            print(f"Episode {ep:03d} {camera}: FAIL (unknown frame count)")
            total_fail += 1
            continue

        print(f"Episode {ep:03d} {camera}: bbox={item['bbox']}, "
              f"frames=[{item['start_frame']},{item['start_frame']+item['num_frames']})")

        # Load frames.
        frames = load_episode_frames(
            item["video_path"], item["start_frame"], item["num_frames"]
        )
        if not frames:
            print(f"  FAIL (no frames loaded)")
            total_fail += 1
            continue
        print(f"  Loaded {len(frames)} frames")

        # Write temp video.
        tmp_dir = args.temp_dir or tempfile.gettempdir()
        tmp_video = Path(tmp_dir) / f"_sam2_ep{ep:03d}_{camera}.mp4"
        if not write_episode_video(frames, tmp_video):
            print(f"  FAIL (cannot write temp video)")
            total_fail += 1
            continue

        # SAM 2.
        masks = generate_masks_sam2(
            tmp_video, item["bbox"], sam2_predictor, args.gaussian_sigma
        )

        # Clean up temp.
        try:
            tmp_video.unlink()
        except OSError:
            pass

        if masks is None:
            print(f"  FAIL (SAM 2 prediction failed)")
            total_fail += 1
            continue

        # Save atomically: write to temp file, then rename.
        # Prevents corrupt NPZ files if the process is killed mid-write.
        # NOTE: np.savez_compressed auto-appends .npz — use a stem that
        # already ends in .npz so the temp file has the right name.
        item["out_file"].parent.mkdir(parents=True, exist_ok=True)
        tmp_file = item["out_file"].with_suffix(".tmp.npz")
        np.savez_compressed(str(tmp_file), masks=masks)
        os.replace(str(tmp_file), str(item["out_file"]))  # atomic on Linux
        size_mb = os.path.getsize(item["out_file"]) / 1024 / 1024
        print(f"  OK  ({masks.shape}, {size_mb:.1f} MB)")
        total_ok += 1

    # ---- Summary ----
    print(f"\nDone.  OK={total_ok}  Skipped={total_skip}  Failed={total_fail}")


if __name__ == "__main__":
    main()
