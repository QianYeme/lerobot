"""Audit C50 8-state data and generate aligned SAM2 masks for a few training episodes."""

import argparse
import hashlib
import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import av
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def episode_rows(root):
    rows = {}
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            episode = int(row["episode_index"])
            if episode in rows:
                raise ValueError(f"Duplicate episode metadata: {episode}")
            rows[episode] = row
    return rows


def boxes(root, episode):
    xml = root / f"annotations/top/episode_{episode:03d}.xml"
    result = {}
    for track in ET.parse(xml).getroot().findall("track"):
        if track.get("label") != "cup":
            continue
        for box in track.findall("box"):
            if box.get("outside", "0") != "0":
                continue
            frame = int(box.get("frame"))
            if frame in result:
                raise ValueError(f"Multiple cup boxes: episode {episode}, frame {frame}")
            result[frame] = [float(box.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")]
    return result, xml


def video_mapping(root, row):
    prefix = "videos/observation.images.top"
    path = root / (
        f"{prefix}/chunk-{int(row[prefix + '/chunk_index']):03d}/"
        f"file-{int(row[prefix + '/file_index']):03d}.mp4"
    )
    return path, float(row[prefix + "/from_timestamp"])


def audit(root, rows):
    info = json.loads((root / "meta/info.json").read_text())
    manifest = json.loads((root / "meta/nomaster_manifest.json").read_text())
    stats = json.loads((root / "meta/stats.json").read_text())
    names = info["features"]["observation.state"]["names"]
    expected = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos",
                "wrist_roll.pos", "gripper.pos", "gripper.load", "gripper.curr"]
    if names != expected or info["features"]["observation.state"]["shape"] != [8]:
        raise ValueError(f"Unexpected state contract: {names}")
    if set(rows) != set(range(50)) or sum(int(r["length"]) for r in rows.values()) != 35900:
        raise ValueError("Expected exactly C50 with 35900 frames")
    if manifest["validation"] != [7, 11, 13, 14, 16, 27, 35, 38, 40, 43]:
        raise ValueError("Validation split changed")
    if set(manifest["train"]) != set(rows) - set(manifest["validation"]):
        raise ValueError("Training / validation overlap or missing episodes")
    for key in ("observation.state", "action"):
        if stats[key]["count"] != [28720]:
            raise ValueError(f"Statistics not computed on train40: {key}")
    source = Path(manifest["source"])
    observed = 0
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["observation.state", "action", "episode_index", "frame_index", "timestamp"])
        original = pq.read_table(source / path.relative_to(root), columns=table.column_names)
        original = original.filter(pc.less(original["episode_index"], 50))
        for key in ("action", "episode_index", "frame_index", "timestamp"):
            if not table[key].equals(original[key]):
                raise ValueError(f"Changed source field: {path}, {key}")
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        old_state = np.asarray(original["observation.state"].to_pylist(), dtype=np.float32)
        if state.shape != (len(table), 8) or not np.isfinite(state).all():
            raise ValueError("Malformed / nonfinite state")
        np.testing.assert_array_equal(state, old_state[:, :8])
        observed += len(table)
    if observed != 35900:
        raise ValueError(f"Unexpected data length: {observed}")
    for ep, row in rows.items():
        labels, _ = boxes(root, ep)
        if set(labels) != set(range(int(row["length"]))):
            raise ValueError(f"Incomplete cup detection labels: episode {ep}")
        video, start = video_mapping(root, row)
        if not video.is_file() or start < 0:
            raise ValueError(f"Invalid video mapping: {video}, {start}")
    return {"status": "passed", "frames": observed, "state_names": names,
            "train_episodes": manifest["train"], "validation_episodes": manifest["validation"],
            "hashes": {str(p.relative_to(root)): sha(p) for p in
                       [root / "meta/info.json", root / "meta/stats.json", root / "meta/nomaster_manifest.json"]},
            "source_parity": "unchanged action / time / indices; state equals source without master"}


def extract_frames(video, start, length, fps, directory):
    """Decode by video-local timestamps, validating every frame instead of silently filling gaps."""
    timestamps = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        container.seek(int(start / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Decoded frame has no timestamp")
            stamp = float(frame.pts * stream.time_base)
            if stamp < start - 0.25 / fps:
                continue
            expected = start + len(timestamps) / fps
            if abs(stamp - expected) > 0.25 / fps:
                raise ValueError(f"Video alignment mismatch: {stamp} vs {expected}")
            Image.fromarray(frame.to_ndarray(format="rgb24")).save(directory / f"{len(timestamps):06d}.jpg", quality=95)
            timestamps.append(stamp)
            if len(timestamps) == length:
                break
    if len(timestamps) != length:
        raise ValueError(f"Decoded {len(timestamps)} frames, expected {length}")
    return timestamps


def generate(root, row, output, predictor, fps, prompt_frames=(0,), negative_margin=0, framewise_box=False):
    import torch

    ep, length = int(row["episode_index"]), int(row["length"])
    labels, xml = boxes(root, ep)
    if 0 not in labels:
        raise ValueError("Frame-zero box is required for this preview generator")
    video, start = video_mapping(root, row)
    mask_file = output / f"masks/top/episode_{ep:03d}.npz"
    record_file = output / f"episode_{ep:03d}.json"
    if mask_file.exists() or record_file.exists():
        raise FileExistsError(f"Refusing to overwrite preview episode {ep}")
    with tempfile.TemporaryDirectory(prefix=f"c50_sam_ep{ep:03d}_", dir=output) as directory:
        frames = Path(directory)
        timestamps = extract_frames(video, start, length, fps, frames)
        with torch.inference_mode():
            state = predictor.init_state(video_path=str(frames), offload_video_to_cpu=True, offload_state_to_cpu=True)
            prompts = []
            def add_prompt(index):
                if index not in labels or index >= length:
                    raise ValueError(f"Missing prompt box at frame {index}")
                box = labels[index]
                points = []
                if negative_margin > 0:
                    x1, y1, x2, y2 = box
                    candidates = [(x1-negative_margin, (y1+y2)/2), (x2+negative_margin, (y1+y2)/2),
                                  ((x1+x2)/2, y1-negative_margin), ((x1+x2)/2, y2+negative_margin)]
                    points = [(x, y) for x, y in candidates if 0 <= x < 640 and 0 <= y < 480]
                kwargs = {}
                if points:
                    kwargs = {"points": np.asarray(points, dtype=np.float32),
                              "labels": np.zeros(len(points), dtype=np.int32)}
                result = predictor.add_new_points_or_box(inference_state=state, frame_idx=index, obj_id=1,
                                                         box=box, **kwargs)
                prompts.append({"frame": index, "box": box, "negative_points": points})
                return result

            def independent_frames():
                for index in range(length):
                    predictor.reset_state(state)
                    yield add_prompt(index)

            if framewise_box:
                predictions = independent_frames()
            else:
                for index in sorted(set(prompt_frames)):
                    add_prompt(index)
                predictions = predictor.propagate_in_video(state)
            masks = np.zeros((length, 480, 640), dtype=np.uint8)
            seen = np.zeros(length, dtype=bool)
            for index, objects, logits in predictions:
                if objects != [1] or index < 0 or index >= length or seen[index]:
                    raise ValueError(f"Unexpected SAM frame or object: {index}, {objects}")
                binary = (logits[0, 0] > 0).cpu().numpy().astype(np.uint8)
                if binary.shape != (480, 640):
                    raise ValueError(f"Unexpected mask size: {binary.shape}")
                masks[index] = binary
                seen[index] = True
                if framewise_box and (index + 1) % 120 == 0:
                    print(json.dumps({"episode": ep, "independent_frames_complete": index + 1}), flush=True)
            if not seen.all():
                raise ValueError(f"SAM propagation omitted frames: {np.flatnonzero(~seen).tolist()}")
            del state
        area = masks.sum(axis=(1, 2)).astype(float)
        flags = {"empty_frames": np.flatnonzero(area == 0).tolist(),
                 "area_jumps_over_50pct": (np.flatnonzero(np.abs(np.diff(area)) > 0.5 * np.maximum(area[:-1], 1)) + 1).tolist()}
        sample_frames = [0, 120, 240, 360, 480, length - 1]
        sheet = Image.new("RGB", (1280, 3 * 510), "white")
        for i, index in enumerate(sample_frames):
            rgb = np.asarray(Image.open(frames / f"{index:06d}.jpg")).copy()
            foreground = masks[index].astype(bool)
            rgb[foreground] = (0.65 * rgb[foreground] + 0.35 * np.array([0, 255, 0])).astype(np.uint8)
            panel = Image.fromarray(rgb)
            draw = ImageDraw.Draw(panel)
            draw.rectangle(labels[index], outline="red", width=2)
            x, y = (i % 2) * 640, (i // 2) * 510
            sheet.paste(panel, (x, y + 30))
            ImageDraw.Draw(sheet).text((x + 8, y + 8), f"episode={ep} frame={index} mask_pixels={int(area[index])}", fill="black")
        sheet.save(output / f"episode_{ep:03d}_preview.jpg", quality=95)
        mask_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(mask_file, masks=masks, valid=seen)
        record = {"episode": ep, "length": length, "camera": "top", "video": str(video),
                  "video_from_timestamp": start, "decoded_timestamps": timestamps,
                  "annotation_sha256": sha(xml), "mask_sha256": sha(mask_file),
                  "label_type": "SAM2 binary pseudo-label; logits > 0; no Gaussian blur",
                  "generation_complete": True, "review_status": "pending", "flags": flags,
                  "prompts": prompts, "negative_margin": negative_margin,
                  "strategy": "independent_frame_box" if framewise_box else "video_propagation",
                  "sample_frames": sample_frames, "foreground_area_min": float(area.min()),
                  "foreground_area_max": float(area.max())}
        record_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.cuda.empty_cache()
    print(json.dumps({"episode": ep, "flags": flags, "mask_file": str(mask_file)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 20, 49])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prompt-frames", type=int, nargs="+", default=[0])
    parser.add_argument("--negative-margin", type=float, default=0)
    parser.add_argument("--framewise-box", action="store_true")
    args = parser.parse_args()
    if 0 not in args.prompt_frames or min(args.prompt_frames) < 0 or args.negative_margin < 0:
        raise ValueError("Prompt frames must include 0 and be nonnegative; margin must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = episode_rows(args.dataset_root)
    report = audit(args.dataset_root, rows)
    if not set(args.episodes).issubset(report["train_episodes"]):
        raise ValueError("Preview episodes must come from training split")
    report["preview_plan"] = [{"episode": ep, "video": str(video_mapping(args.dataset_root, rows[ep])[0]),
                               "start_timestamp": video_mapping(args.dataset_root, rows[ep])[1],
                               "length": int(rows[ep]["length"])} for ep in args.episodes]
    (args.output / "dataset_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"audit": report["status"], "frames": report["frames"], "plan": report["preview_plan"]}), flush=True)
    if args.dry_run:
        return
    if args.checkpoint is None or not args.checkpoint.is_file():
        raise FileNotFoundError("A SAM2 checkpoint is required")
    from sam2.build_sam import build_sam2_video_predictor
    import torch

    torch.set_num_threads(4)

    predictor = build_sam2_video_predictor(args.sam_config, str(args.checkpoint), device="cuda", apply_postprocessing=False)
    fps = json.loads((args.dataset_root / "meta/info.json").read_text())["fps"]
    for ep in args.episodes:
        generate(args.dataset_root, rows[ep], args.output, predictor, fps, args.prompt_frames,
                 args.negative_margin, args.framewise_box)
    (args.output / "generation.done").write_text("generation_complete; review_pending\n", encoding="utf-8")


if __name__ == "__main__":
    main()
