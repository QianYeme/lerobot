"""Build per-episode phase-boundary contact sheets and a review CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


def read_frame(video: cv2.VideoCapture, index: int):
    video.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = video.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {index}")
    return frame


def build_pack(annotations_path: Path, videos: Path, output: Path, reviewed: set[int]) -> None:
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    sheets = output / "contact_sheets"
    sheets.mkdir(exist_ok=True)
    rows = []

    for episode_text, record in annotations["episodes"].items():
        episode = int(episode_text)
        events = record["events"]
        close = int(events["close"])
        release = int(events["release"])
        grasp_stop = min(release, close + int(annotations["config"]["hold_frames"]))
        frame_indices = [max(0, close - 1), close, grasp_stop, release]
        labels = ["before_close", "close", "lift_start", "release"]
        video_path = videos / f"episode_{episode:03d}.mp4"
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open {video_path}")
        panels = []
        try:
            for label, frame_index in zip(labels, frame_indices, strict=True):
                frame = read_frame(capture, frame_index)
                cv2.putText(
                    frame,
                    f"ep{episode:03d} {label} f{frame_index}",
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                panels.append(frame)
        finally:
            capture.release()
        sheet = cv2.hconcat(panels)
        sheet_path = sheets / f"episode_{episode:03d}.jpg"
        if not cv2.imwrite(str(sheet_path), sheet):
            raise RuntimeError(f"Could not write {sheet_path}")
        rows.append(
            {
                "episode_index": episode,
                "close_frame": close,
                "lift_start_frame": grasp_stop,
                "release_frame": release,
                "load_diagnostic": "load_contact_mismatch_diagnostic" in record["warnings"],
                "review_status": "approved" if episode in reviewed else "pending",
                "review_note": "user-confirmed" if episode in reviewed else "",
                "contact_sheet": str(sheet_path.relative_to(output)),
            }
        )

    with (output / "review.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewed", type=int, nargs="*", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_pack(args.annotations, args.videos, args.output, set(args.reviewed))
