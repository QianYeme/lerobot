#!/usr/bin/env python
"""Freeze and export the train-only P0 reference/check frame selection."""

import argparse
import json
from pathlib import Path

import cv2
import pyarrow.parquet as pq


REFERENCE_EPISODES = [0, 6, 10, 17, 22, 35, 40, 55]
CHECK_EPISODES = [2, 7, 13, 18, 23, 25, 30, 36, 42, 45, 50, 56]
FRAME_INDEX = 15


def read_frame(video: Path, timestamp_s: float):
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Failed to read {video} at {timestamp_s:.3f}s")
    return frame


def main(args):
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    split_by_episode = {row["episode_index"]: row for row in split["episodes"]}
    selected = REFERENCE_EPISODES + CHECK_EPISODES
    if any(split_by_episode[index]["split"] != "train" for index in selected):
        raise RuntimeError("P0 selection contains a non-train episode.")

    episode_table = pq.read_table(
        args.dataset / "meta/episodes/chunk-000/file-000.parquet"
    ).to_pylist()
    episode_by_index = {row["episode_index"]: row for row in episode_table}
    images_dir = args.output / "selected_frames"
    images_dir.mkdir(parents=True, exist_ok=True)
    records = []
    thumbnails = []
    for subset, episodes in (("reference", REFERENCE_EPISODES), ("calibration_check", CHECK_EPISODES)):
        for episode_index in episodes:
            row = episode_by_index[episode_index]
            video_file = row["videos/observation.images.top/file_index"]
            timestamp = row["videos/observation.images.top/from_timestamp"] + FRAME_INDEX / 30
            video = (
                args.dataset
                / "videos/observation.images.top/chunk-000"
                / f"file-{video_file:03d}.mp4"
            )
            frame = read_frame(video, timestamp)
            filename = f"{subset}_ep{episode_index:03d}_frame{FRAME_INDEX:03d}.png"
            cv2.imwrite(str(images_dir / filename), frame)
            split_row = split_by_episode[episode_index]
            records.append(
                {
                    "episode_index": episode_index,
                    "frame_index": FRAME_INDEX,
                    "source_split": "train",
                    "subset": subset,
                    "stratum": split_row["stratum"],
                    "source": split_row["source"],
                    "selection_reason": "pre-frozen spatial-stratum coverage; selected before DINO diagnostics",
                    "dataset_key": "observation.images.top",
                    "image_path": f"selected_frames/{filename}",
                    "video_file_index": video_file,
                    "video_timestamp_s": timestamp,
                }
            )
            thumb = cv2.resize(frame, (320, 240))
            cv2.rectangle(thumb, (0, 0), (320, 30), (0, 0, 0), -1)
            cv2.putText(
                thumb,
                f"{subset[:3]} ep{episode_index} {split_row['stratum']}",
                (7, 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            thumbnails.append(thumb)

    manifest = {
        "version": 1,
        "selection_frozen_before_feature_extraction": True,
        "frame_index_within_episode": FRAME_INDEX,
        "reference_count": len(REFERENCE_EPISODES),
        "calibration_check_count": len(CHECK_EPISODES),
        "forbidden_splits_read": [],
        "records": records,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "reference_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    rows = []
    for start in range(0, len(thumbnails), 4):
        row = thumbnails[start : start + 4]
        while len(row) < 4:
            row.append(255 * row[0].copy())
        rows.append(cv2.hconcat(row))
    cv2.imwrite(str(args.output / "selection_contact_sheet.png"), cv2.vconcat(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
