"""Serve the local four-step phase review UI and persist its decisions."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


STEPS = ("closing", "lift_start", "release")


def parse_byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    if header is None:
        return None
    if not header.startswith("bytes=") or "," in header:
        raise ValueError("Only a single byte range is supported")
    start_text, separator, end_text = header.removeprefix("bytes=").partition("-")
    if not separator or (not start_text and not end_text):
        raise ValueError("Invalid byte range")
    if start_text:
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
    else:
        suffix = int(end_text)
        if suffix <= 0:
            raise ValueError("Invalid suffix range")
        start = max(0, size - suffix)
        end = size - 1
    if start < 0 or start >= size or end < start:
        raise ValueError("Byte range is outside the file")
    return start, min(end, size - 1)


def initial_reviews(annotations: dict) -> dict[str, dict]:
    reviews = {}
    for episode_text, record in annotations["episodes"].items():
        episode = int(episode_text)
        events = record["events"]
        reviews[str(episode)] = {
            "episode_index": episode,
            "length": int(record["length"]),
            "load_diagnostic": "load_contact_mismatch_diagnostic" in record["warnings"],
            "reviewed": False,
            "note": "",
            "steps": {
                "closing": {
                    "start_frame": int(events["close_motion_start"]),
                    "end_frame": int(events["close_motion_end"]),
                    "note": "",
                },
                "lift_start": {
                    "frame": int(events["close_motion_end"]) + 1,
                    "note": "",
                },
                "release": {
                    "start_frame": int(events["release_motion_start"]),
                    "end_frame": int(events["release_motion_end"]),
                    "note": "",
                },
            },
        }
    return reviews


def validate_reviews(payload: dict, episode_lengths: dict[int, int]) -> dict:
    episodes = payload.get("episodes")
    if not isinstance(episodes, dict) or set(map(int, episodes)) != set(episode_lengths):
        raise ValueError("Review payload must contain every dataset episode exactly once")
    for episode_text, review in episodes.items():
        episode = int(episode_text)
        steps = review.get("steps", {})
        if set(steps) != set(STEPS):
            raise ValueError(f"Episode {episode} must contain exactly four review steps")
        closing = steps["closing"]
        lift = steps["lift_start"]
        release = steps["release"]
        frame_fields = (
            ("closing.start_frame", closing.get("start_frame")),
            ("closing.end_frame", closing.get("end_frame")),
            ("lift_start.frame", lift.get("frame")),
            ("release.start_frame", release.get("start_frame")),
            ("release.end_frame", release.get("end_frame")),
        )
        frames = []
        for name, frame in frame_fields:
            if not isinstance(frame, int) or not 0 <= frame < episode_lengths[episode]:
                raise ValueError(f"Episode {episode} {name} has an invalid frame")
            frames.append(frame)
        if not isinstance(review.get("reviewed"), bool):
            raise ValueError(f"Episode {episode} reviewed must be boolean")
        if frames != sorted(frames):
            raise ValueError(f"Episode {episode} interval boundaries must be nondecreasing")
    return payload


def normalize_browser_payload(payload: dict) -> dict:
    """Convert an in-memory schema-2 review into the schema-3 reviewed format."""
    if payload.get("schema_version") == 3:
        return payload
    normalized = {"schema_version": 3, "episodes": payload["episodes"]}
    for review in normalized["episodes"].values():
        review["reviewed"] = True
        review.setdefault("note", "")
        for step in review["steps"].values():
            step.pop("status", None)
            step.setdefault("note", "")
    return normalized


def write_reviews(payload: dict, json_path: Path, csv_path: Path) -> None:
    json_temp = json_path.with_suffix(".json.tmp")
    json_temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    json_temp.replace(json_path)
    csv_temp = csv_path.with_suffix(".csv.tmp")
    with csv_temp.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["episode_index", "reviewed", "episode_note", "closing_start_frame", "closing_end_frame", "closing_note", "lift_start_frame", "lift_start_note", "release_start_frame", "release_end_frame", "release_note"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for episode_text in sorted(payload["episodes"], key=int):
            review = payload["episodes"][episode_text]
            row = {"episode_index": int(episode_text)}
            closing = review["steps"]["closing"]
            lift = review["steps"]["lift_start"]
            release = review["steps"]["release"]
            row.update({
                "reviewed": review["reviewed"], "episode_note": review.get("note", ""),
                "closing_start_frame": closing["start_frame"], "closing_end_frame": closing["end_frame"], "closing_note": closing.get("note", ""),
                "lift_start_frame": lift["frame"], "lift_start_note": lift.get("note", ""),
                "release_start_frame": release["start_frame"], "release_end_frame": release["end_frame"], "release_note": release.get("note", ""),
            })
            writer.writerow(row)
    csv_temp.replace(csv_path)


def make_handler(ui_dir: Path, video_dir: Path, bootstrap: dict, output_dir: Path):
    lengths = {int(key): int(value["length"]) for key, value in bootstrap["episodes"].items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "phase_step_reviews.json"
    csv_path = output_dir / "phase_step_reviews.csv"

    class Handler(SimpleHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def translate_path(self, path: str) -> str:
            route = unquote(urlparse(path).path)
            if route.startswith("/videos/"):
                name = Path(route.removeprefix("/videos/")).name
                return str(video_dir / name)
            relative = route.lstrip("/") or "index.html"
            return str(ui_dir / relative)

        def do_GET(self) -> None:
            if urlparse(self.path).path == "/api/bootstrap":
                body = json.dumps(bootstrap, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if urlparse(self.path).path.startswith("/videos/"):
                self.serve_video(send_body=True)
                return
            super().do_GET()

        def do_HEAD(self) -> None:
            if urlparse(self.path).path.startswith("/videos/"):
                self.serve_video(send_body=False)
                return
            super().do_HEAD()

        def serve_video(self, *, send_body: bool) -> None:
            path = Path(self.translate_path(self.path))
            if not path.is_file():
                self.send_error(404, "Video not found")
                return
            size = path.stat().st_size
            try:
                byte_range = parse_byte_range(self.headers.get("Range"), size)
            except (ValueError, TypeError):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            start, end = byte_range if byte_range is not None else (0, size - 1)
            self.send_response(206 if byte_range is not None else 200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if byte_range is not None:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if not send_body:
                return
            remaining = end - start + 1
            with path.open("rb") as stream:
                stream.seek(start)
                while remaining:
                    block = stream.read(min(64 * 1024, remaining))
                    if not block:
                        break
                    try:
                        self.wfile.write(block)
                    except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                        return
                    remaining -= len(block)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/save":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                payload = json.loads(raw)
                if payload.get("schema_version") != 3:
                    rescue = output_dir / f"browser_rescue_raw_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
                    rescue.write_bytes(raw)
                payload = normalize_browser_payload(payload)
                validate_reviews(payload, lengths)
                write_reviews(payload, json_path, csv_path)
                body = b'{"ok":true}'
                self.send_response(200)
            except (ValueError, json.JSONDecodeError) as error:
                body = json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False).encode("utf-8")
                self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            print(f"[phase-review] {format % args}")

    return Handler


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=root / "outputs/formal3_phase_labels_20260927/export_v3_intervals_review/phase_annotations.json")
    parser.add_argument("--videos", type=Path, default=root / "数据集/formal3/kind_merged/annotation_segments_crf18/videos/observation.images.top")
    parser.add_argument("--output", type=Path, default=root / "outputs/formal3_phase_labels_20260927/review_ui_v3_reviewed")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations = json.loads(args.annotations.resolve().read_text(encoding="utf-8"))
    output = args.output.resolve()
    saved = output / "phase_step_reviews.json"
    if saved.exists():
        reviews = validate_reviews(json.loads(saved.read_text(encoding="utf-8")), {int(k): int(v["length"]) for k, v in annotations["episodes"].items()})["episodes"]
    else:
        reviews = initial_reviews(annotations)
    bootstrap = {"schema_version": 3, "fps": float(annotations["dataset"]["fps"]), "episodes": reviews}
    ui_dir = Path(__file__).resolve().parents[1] / "tools/phase_review_ui"
    handler = make_handler(ui_dir, args.videos.resolve(), bootstrap, output)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Phase review UI: http://{args.host}:{args.port}")
    print(f"Reviews save to: {output}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
