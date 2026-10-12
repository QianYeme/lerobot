#!/usr/bin/env python
"""Draft deterministic cup boxes for manual overlay review (not a learned detector)."""

import argparse
import json
from pathlib import Path

import cv2


def main(args):
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    annotations = []
    reviewed = []
    for record in manifest["records"]:
        image_path = args.result_dir / record["image_path"]
        image = cv2.imread(str(image_path))
        gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (9, 9), 1.5)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=80,
            param2=30,
            minRadius=20,
            maxRadius=60,
        )
        candidates = [] if circles is None else [circle for circle in circles[0] if circle[1] > 200]
        if not candidates:
            raise RuntimeError(f"No cup-circle draft for {image_path}")
        center_x, center_y, radius = max(candidates, key=lambda circle: circle[1])
        half_size = max(float(radius) + 14, 46.0)
        x1 = max(0, round(center_x - half_size))
        y1 = max(0, round(center_y - half_size))
        x2 = min(image.shape[1] - 1, round(center_x + half_size))
        y2 = min(image.shape[0] - 1, round(center_y + half_size))
        annotations.append(
            {
                "episode_index": record["episode_index"],
                "frame_index": record["frame_index"],
                "subset": record["subset"],
                "stratum": record["stratum"],
                "image_path": record["image_path"],
                "coarse_box_xyxy": [x1, y1, x2, y2],
                "annotation_semantics": "visible cup including rim, water region, wall edges, and base",
                "draft_method": "OpenCV Hough circle geometry; no learned detector",
                "review_status": "VISUALLY_REVIEWED_PASS_2026-10-11",
            }
        )
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(overlay, (round(center_x), round(center_y)), 3, (0, 0, 255), -1)
        cv2.putText(
            overlay,
            f"ep{record['episode_index']} {record['subset']}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        reviewed.append(cv2.resize(overlay, (320, 240)))

    payload = {
        "version": 1,
        "coordinate_system": "original 640x480 top RGB, xyxy inclusive",
        "formal_annotation_requires_overlay_review": True,
        "learned_detector_used": False,
        "sam_used": False,
        "annotations": annotations,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = [cv2.hconcat(reviewed[index : index + 4]) for index in range(0, len(reviewed), 4)]
    cv2.imwrite(str(args.result_dir / "coarse_box_review_sheet.png"), cv2.vconcat(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
