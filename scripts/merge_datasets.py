#!/usr/bin/env python3
"""Merge two LeRobot sub-datasets (e.g. kind1 + kind2) into one.

Videos are concatenated with ``ffmpeg -c copy`` (no re-encode).
Episodes from kind2 are renumbered to continue after kind1.
Annotations (XML) and masks (NPZ) are copied with updated episode indices.

Training: LeRobot loads all episodes and shuffles frames randomly,
so storage order does not affect training.

Train/test split: if you want a 70/30 split that represents BOTH kinds,
you can manually select episodes:
    kind1 0-34 + kind2 50-77 → train (63 episodes)
    kind1 35-49 + kind2 78-89 → test (27 episodes)

Usage:
    python scripts/merge_datasets.py \\
        数据集/formal/kind1 数据集/formal/kind2 数据集/formal_A
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def merge_datasets(src1: Path, src2: Path, dst: Path):
    """Merge two LeRobot v3 datasets into one."""
    if dst.exists():
        print(f"[ERROR] Output exists: {dst}. Remove first.")
        sys.exit(1)

    print(f"Merging:\n  {src1}\n  {src2}\n  → {dst}\n")

    # ---- Validate ----
    info1 = json.loads((src1 / "meta/info.json").read_text())
    info2 = json.loads((src2 / "meta/info.json").read_text())

    dim1 = info1["features"]["observation.state"]["shape"][0]
    dim2 = info2["features"]["observation.state"]["shape"][0]
    assert dim1 == dim2, f"State dim mismatch: {dim1} vs {dim2}"
    assert info1["fps"] == info2["fps"], "FPS mismatch"

    ep1, ep2 = info1["total_episodes"], info2["total_episodes"]
    total_ep = ep1 + ep2
    total_frames = info1["total_frames"] + info2["total_frames"]
    max_ep1 = ep1 - 1  # kind1 episodes: 0 .. max_ep1

    print(f"  kind1: {ep1} episodes, {info1['total_frames']} frames")
    print(f"  kind2: {ep2} episodes, {info2['total_frames']} frames")
    print(f"  merged: {total_ep} episodes, {total_frames} frames\n")

    dst.mkdir(parents=True, exist_ok=True)

    # ---- Merge data parquet ----
    print("Merging data parquet...")
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pandas as pd

    data1 = pq.read_table(src1 / "data/chunk-000/file-000.parquet").to_pandas()
    data2 = pq.read_table(src2 / "data/chunk-000/file-000.parquet").to_pandas()

    # Renumber kind2 episodes.
    data2["episode_index"] = data2["episode_index"] + ep1

    # Normalize column types (kind1 may use fixed_size_list, kind2 may use list).
    for col in ["action", "observation.state"]:
        if col in data1.columns:
            data1[col] = data1[col].apply(lambda x: list(x) if hasattr(x, 'tolist') else x)
            data2[col] = data2[col].apply(lambda x: list(x) if hasattr(x, 'tolist') else x)

    merged = pd.concat([data1, data2], ignore_index=True)
    os.makedirs(dst / "data" / "chunk-000", exist_ok=True)
    pq.write_table(pa.Table.from_pandas(merged),
                   str(dst / "data/chunk-000/file-000.parquet"))
    print(f"  data: {len(data1)} + {len(data2)} = {len(merged)} rows")

    # ---- Merge meta ----
    print("Merging meta...")
    os.makedirs(dst / "meta" / "episodes" / "chunk-000", exist_ok=True)

    ep1_meta = pq.read_table(src1 / "meta/episodes/chunk-000/file-000.parquet").to_pandas()
    ep2_meta = pq.read_table(src2 / "meta/episodes/chunk-000/file-000.parquet").to_pandas()
    ep2_meta["episode_index"] = ep2_meta["episode_index"] + ep1

    # Update data offsets for kind2.
    src1_offset = info1["total_frames"]
    for col in ["dataset_from_index", "dataset_to_index"]:
        if col in ep2_meta.columns:
            ep2_meta[col] = ep2_meta[col] + src1_offset

    merged_ep = pd.concat([ep1_meta, ep2_meta], ignore_index=True)
    pq.write_table(pa.Table.from_pandas(merged_ep),
                   str(dst / "meta/episodes/chunk-000/file-000.parquet"))

    for f in ["tasks.parquet", "stats.json"]:
        src_f = src1 / "meta" / f
        if src_f.exists():
            shutil.copy2(src_f, dst / "meta" / f)

    # info.json
    merged_info = dict(info1)
    merged_info["total_episodes"] = total_ep
    merged_info["total_frames"] = total_frames
    # Note: kind1=0..{max_ep1}, kind2={ep1}..{total_ep-1}
    # For a 70/30 split mixing both kinds, manually select:
    #   train: kind1[0:35] + kind2[{ep1}:{ep1+28}]
    #   test:  kind1[35:50] + kind2[{ep1+28}:{total_ep}]
    merged_info["splits"] = {
        "train": f"0:{total_ep}",
        "kind1_range": f"0:{ep1}",
        "kind2_range": f"{ep1}:{total_ep}",
    }
    (dst / "meta/info.json").write_text(json.dumps(merged_info, indent=2))
    print("  meta: done")

    # ---- Merge videos (simple concat) ----
    print("Merging videos...")
    video_keys = [k for k in info1["features"].keys()
                  if k.startswith("observation.images.")]

    for vk in video_keys:
        v1 = src1 / "videos" / vk / "chunk-000" / "file-000.mp4"
        v2 = src2 / "videos" / vk / "chunk-000" / "file-000.mp4"
        vd = dst / "videos" / vk / "chunk-000"
        os.makedirs(vd, exist_ok=True)

        if v1.exists() and v2.exists():
            concat_list = vd / "concat.txt"
            concat_list.write_text(
                f"file '{v1.absolute()}'\nfile '{v2.absolute()}'\n"
            )
            subprocess.run([
                "ffmpeg", "-f", "concat", "-safe", "0",
                "-i", str(concat_list), "-c", "copy",
                "-y", "-nostdin",
                str(vd / "file-000.mp4"),
            ], capture_output=True)
            concat_list.unlink()
            size_mb = os.path.getsize(vd / "file-000.mp4") / 1024 / 1024
            print(f"  {vk}: {size_mb:.0f} MB (concat)")
        elif v1.exists():
            shutil.copy2(v1, vd / "file-000.mp4")
        elif v2.exists():
            shutil.copy2(v2, vd / "file-000.mp4")

    # ---- Merge annotations ----
    print("Merging annotations...")
    for camera_dir in ["top", "gripper"]:
        dst_ann = dst / "annotations" / camera_dir
        os.makedirs(dst_ann, exist_ok=True)

        # Copy kind1 XML (keep original episode indices).
        src1_ann = src1 / "annotations" / camera_dir
        if src1_ann.exists():
            for f in src1_ann.glob("*.xml"):
                shutil.copy2(f, dst_ann / f.name)

        # Copy kind2 XML (renumbered).
        src2_ann = src2 / "annotations" / camera_dir
        if src2_ann.exists():
            for f in sorted(src2_ann.glob("*.xml")):
                try:
                    old_idx = int(f.stem.split("_")[-1])
                    new_name = f"episode_{old_idx + ep1:03d}.xml"
                    shutil.copy2(f, dst_ann / new_name)
                except (ValueError, IndexError):
                    shutil.copy2(f, dst_ann / f.name)

        # Copy kind1 NPZ masks.
        dst_masks = dst / "annotations" / "masks" / camera_dir
        os.makedirs(dst_masks, exist_ok=True)

        src1_masks = src1 / "annotations" / "masks" / camera_dir
        if src1_masks.exists():
            for f in src1_masks.glob("*.npz"):
                shutil.copy2(f, dst_masks / f.name)

        # Copy kind2 NPZ masks (renumbered).
        src2_masks = src2 / "annotations" / "masks" / camera_dir
        if src2_masks.exists():
            for f in sorted(src2_masks.glob("*.npz")):
                try:
                    old_idx = int(f.stem.split("_")[-1])
                    new_name = f"episode_{old_idx + ep1:03d}.npz"
                    shutil.copy2(f, dst_masks / new_name)
                except (ValueError, IndexError):
                    shutil.copy2(f, dst_masks / f.name)

        n_xml = len(list(dst_ann.glob("*.xml")))
        n_npz = len(list(dst_masks.glob("*.npz")))
        print(f"  {camera_dir}: {n_xml} XML, {n_npz} NPZ")

    print(f"\nDone: {dst}")
    print(f"  Episodes: {total_ep}")
    print(f"    kind1: 0-{max_ep1}  ({ep1} episodes)")
    print(f"    kind2: {ep1}-{total_ep-1}  ({ep2} episodes)")
    print(f"  Frames: {total_frames}")
    print(f"  State dim: {dim1}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python scripts/merge_datasets.py <src1> <src2> <dst>")
        sys.exit(1)
    merge_datasets(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
