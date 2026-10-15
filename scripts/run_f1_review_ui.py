"""Loopback-only F1 RGB review; save independent intervals directly to the existing CSV."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

if __package__:
    from .run_phase_review_ui import parse_byte_range
else:
    from run_phase_review_ui import parse_byte_range

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / 'experiments/force/f1_formal4_signal_feasibility/index1/result'
FIELDS = ['episode', 'label', 'first_reliable_frame', 'last_reliable_frame', 'uncertain_start',
          'uncertain_end', 'reviewer', 'evidence', 'reason']
LABELS = ['PRECONTACT', 'CONTACT_SUPPORTED', 'CLOSING', 'POST_CLOSE', 'RELEASE', 'UNKNOWN', 'OCCLUDED']
FRAME_FIELDS = FIELDS[2:6]


class ConflictError(ValueError):
    pass


class ReviewStore:
    def __init__(self, result_dir, cache_dir):
        self.result = Path(result_dir).resolve()
        self.csv_path = self.result / 'evidence_samples/review_intervals.csv'
        manifest = json.loads((self.result / 'label_manifest.json').read_text(encoding='utf-8'))
        self.records = {int(r['episode']): r for r in manifest['records']}
        self.pilot = manifest.get('experiment_id') == 'F1P_CONTACT_SIGNAL_PILOT'
        self.labels = ['PRECONTACT', 'CONTACT_SUPPORTED', 'UNKNOWN', 'OCCLUDED'] if self.pilot else LABELS
        expected = [0, 2, 4, 6, 7, 9]
        if self.pilot:
            split = json.loads((ROOT / 'experiments/data/d1_training_readiness/index1/result/split_manifest_v1.json').read_text())
            train = [r['episode_index'] for r in split['episodes'] if r['split'] == 'train']
            remaining = [ep for ep in train if ep not in expected]
            if len(remaining) != 30:
                raise ValueError('Pilot source Train manifest has changed')
            expected = [remaining[i] for i in [0, 5, 11, 17, 23, 29]]
            subset = json.loads((self.result / 'subset_manifest.json').read_text())
            if subset['pilot_eval'] != expected or subset['pilot_fit'] != [0, 2, 4, 6, 7, 9]:
                raise ValueError('Pilot subset differs from frozen deterministic selection')
        if list(self.records) != expected or any(r['split'] != 'train' for r in self.records.values()):
            raise ValueError('UI episodes differ from the frozen Train-only review subset')
        self.sources = {}
        for ep, record in self.records.items():
            path = (self.result / record['clip']).resolve()
            if not path.is_relative_to(self.result / 'evidence_samples') or not path.is_file():
                raise ValueError('Video must be an existing blind evidence file')
            self.sources[ep] = path
        self.cache = Path(cache_dir).resolve()
        self.videos = {}
        self.lock = threading.Lock()

    def revision(self):
        return hashlib.sha256(self.csv_path.read_bytes()).hexdigest()

    def read(self):
        with self.csv_path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != FIELDS:
                raise ValueError('CSV columns differ from F1 interval schema; will not overwrite')
            rows = list(reader)
        for row in rows:
            if int(row['episode']) not in self.records:
                raise ValueError('CSV contains episodes outside first-six gate; will not overwrite')
        return rows

    def bootstrap(self):
        with self.lock:
            rows = self.read()
            return {'episodes': [{'episode': ep, 'frames': r['frames'], 'fps': r['fps'],
                                  'video': f'/videos/{ep}', 'rows': [x for x in rows if int(x['episode']) == ep and x['label']]}
                                 for ep, r in self.records.items()],
                    'revision': self.revision(), 'csv_path': str(self.csv_path), 'labels': self.labels,
                    'autosave': self.pilot, 'pilot': self.pilot}

    def validate(self, ep, rows):
        if type(ep) is not int or ep not in self.records or not isinstance(rows, list) or len(rows) > 200:
            raise ValueError('Invalid episode or interval list')
        cleaned = []
        for raw in rows:
            if not isinstance(raw, dict):
                raise ValueError('Each interval must be an object')
            row = {k: str(raw.get(k, '')).strip() for k in FIELDS}
            row['episode'] = str(ep)
            if row['label'] not in self.labels or not row['reviewer']:
                raise ValueError('每段必须选择标签并填写审核者')
            for key in FRAME_FIELDS:
                value = row[key]
                if value and (not value.isascii() or not value.isdigit() or not 0 <= int(value) < self.records[ep]['frames']):
                    raise ValueError(f'{key}: 帧号必须为视频范围内的整数')
            for start, end in [('first_reliable_frame', 'last_reliable_frame'), ('uncertain_start', 'uncertain_end')]:
                if bool(row[start]) != bool(row[end]) or (row[start] and int(row[start]) > int(row[end])):
                    raise ValueError('每个区间需要起止两帧，且开始不能大于结束')
            unknown = row['label'] in ('UNKNOWN', 'OCCLUDED')
            if unknown:
                if row['first_reliable_frame'] or not row['uncertain_start'] or not row['reason']:
                    raise ValueError('不可判/遮挡只能填写不确定区间，并必须说明原因')
            elif not row['first_reliable_frame'] or not row['reason']:
                raise ValueError('可靠标签必须填写可靠起止帧及可见证据说明')
            if row['first_reliable_frame'] and row['uncertain_start']:
                a, b, c, d = [int(row[k]) for k in FRAME_FIELDS]
                if max(a, c) <= min(b, d):
                    raise ValueError('可靠区间不得包含不确定帧；请拆成多段')
            row['evidence'] = self.sources[ep].name + '#frames=' + (
                f"{row['uncertain_start']}-{row['uncertain_end']}" if unknown else
                f"{row['first_reliable_frame']}-{row['last_reliable_frame']}")
            cleaned.append(row)
        # Contact and motion-phase are different axes; their overlap is explicitly allowed.
        for i, a in enumerate(cleaned):
            for b in cleaned[i + 1:]:
                for group in ({'PRECONTACT', 'CONTACT_SUPPORTED'}, {'CLOSING', 'POST_CLOSE', 'RELEASE'}):
                    if a['label'] in group and b['label'] in group and a['label'] != b['label']:
                        if max(int(a['first_reliable_frame']), int(b['first_reliable_frame'])) <= min(int(a['last_reliable_frame']), int(b['last_reliable_frame'])):
                            raise ValueError('同一标签轴的不同可靠类别不能重叠')
        return cleaned

    def save(self, ep, rows, revision):
        rows = self.validate(ep, rows)
        with self.lock:
            if revision != self.revision():
                raise ConflictError('CSV已被其他页面或程序更新；先下载草稿，再重新载入，不会覆盖新记录')
            existing = self.read()
            replacement = rows or [{**dict.fromkeys(FIELDS, ''), 'episode': str(ep), 'evidence': self.sources[ep].name}]
            updated = []
            for index in self.records:
                updated.extend(replacement if index == ep else [r for r in existing if int(r['episode']) == index])
            backup_dir = self.csv_path.parent / 'csv_backups'
            backup_dir.mkdir(exist_ok=True)
            backup = backup_dir / ('review_intervals_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.csv')
            shutil.copy2(self.csv_path, backup)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8-sig', newline='',
                                                 dir=self.csv_path.parent, suffix='.tmp', delete=False) as stream:
                    temporary = Path(stream.name)
                    writer = csv.DictWriter(stream, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(updated)
                temporary.replace(self.csv_path)
            finally:
                if temporary and temporary.exists():
                    temporary.unlink()
            readback = [r for r in self.read() if int(r['episode']) == ep and r['label']]
            if readback != rows:
                raise OSError('Saved CSV readback differs; backup retained')
            return {'ok': True, 'revision': self.revision(), 'rows': readback,
                    'csv_path': str(self.csv_path), 'backup': str(backup)}

    def prepare_videos(self):
        """Original OpenCV mp4v clips are not reliably browser-decodable; cache H264 faststart."""
        self.cache.mkdir(parents=True, exist_ok=True)
        for ep, source in self.sources.items():
            stamp = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
            dest = self.cache / f'ep{ep:03d}_{stamp}.mp4'
            if not dest.exists():
                temp = dest.with_suffix('.pending.mp4')
                print(f'Preparing browser video Train {ep}...', flush=True)
                subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(source), '-an',
                                '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
                                '-movflags', '+faststart', str(temp)], check=True)
                import cv2
                cap = cv2.VideoCapture(str(temp))
                count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if count != self.records[ep]['frames'] or abs(fps - self.records[ep]['fps']) > .01:
                    raise ValueError('Browser transcode changed frame count/FPS')
                temp.replace(dest)
            self.videos[ep] = dest


def make_handler(store):
    ui = ROOT / 'tools/f1_review_ui'

    class Handler(BaseHTTPRequestHandler):
        def trusted_host(self):
            return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',
                                                f'localhost:{self.server.server_port}')

        def respond(self, status, body, kind='application/json; charset=utf-8', headers=None, head=False):
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if not head:
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

        def do_HEAD(self):
            self.do_GET(head=True)

        def do_GET(self, head=False):
            if not self.trusted_host():
                return self.respond(403, b'{}', head=head)
            parsed = urlparse(self.path)
            route = parsed.path
            try:
                if route == '/api/bootstrap':
                    return self.respond(200, json.dumps(store.bootstrap(), ensure_ascii=False).encode(), head=head)
                if route in ('/', '/app.js', '/styles.css'):
                    name, kind = {'/': ('index.html', 'text/html; charset=utf-8'),
                                  '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                                  '/styles.css': ('styles.css', 'text/css; charset=utf-8')}[route]
                    return self.respond(200, (ui / name).read_bytes(), kind, head=head)
                if route.startswith('/videos/'):
                    ep = int(route.split('/')[-1])
                    path = store.videos[ep]
                    size = path.stat().st_size
                    try:
                        selected = parse_byte_range(self.headers.get('Range'), size)
                    except ValueError:
                        return self.respond(416, b'', headers={'Content-Range': f'bytes */{size}'}, head=head)
                    start, end = selected or (0, size - 1)
                    with path.open('rb') as stream:
                        stream.seek(start)
                        body = stream.read(end - start + 1)
                    headers = {'Accept-Ranges': 'bytes'}
                    if selected:
                        headers['Content-Range'] = f'bytes {start}-{end}/{size}'
                    return self.respond(206 if selected else 200, body, 'video/mp4', headers, head)
                if route == '/api/frame':
                    args = parse_qs(parsed.query)
                    ep, frame = int(args['episode'][0]), int(args['frame'][0])
                    if not 0 <= frame < store.records[ep]['frames']:
                        raise ValueError('Frame outside episode')
                    import cv2
                    cap = cv2.VideoCapture(str(store.sources[ep]))
                    try:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                        ok, image = cap.read()
                    finally:
                        cap.release()
                    if not ok:
                        raise ValueError('Frame decode failed')
                    ok, data = cv2.imencode('.jpg', image)
                    if not ok:
                        raise ValueError('JPEG encode failed')
                    return self.respond(200, data.tobytes(), 'image/jpeg', head=head)
                return self.respond(404, b'{}')
            except (ValueError, KeyError) as error:
                return self.respond(400, json.dumps({'error': str(error)}).encode(), head=head)

        def do_POST(self):
            if not self.trusted_host():
                return self.respond(403, b'{}')
            if self.path != '/api/save':
                return self.respond(404, b'{}')
            # Loopback binding plus exact same-origin JSON protects local file writes.
            origin = self.headers.get('Origin')
            if origin and origin != f'http://{self.headers.get("Host")}':
                return self.respond(403, b'{"error":"Cross-origin write rejected"}')
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self.respond(415, b'{}')
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1_000_000:
                    raise ValueError('Invalid request length')
                body = json.loads(self.rfile.read(length))
                result = store.save(body['episode'], body['rows'], body['revision'])
                self.respond(200, json.dumps(result, ensure_ascii=False).encode())
            except ConflictError as error:
                self.respond(409, json.dumps({'error': str(error)}, ensure_ascii=False).encode())
            except (ValueError, KeyError, TypeError) as error:
                self.respond(400, json.dumps({'error': str(error)}, ensure_ascii=False).encode())
            except OSError as error:
                self.respond(500, json.dumps({'error': str(error)}, ensure_ascii=False).encode())

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, default=DEFAULT_RESULT)
    parser.add_argument('--cache', type=Path, default=ROOT / 'outputs/f1_rgb_review_ui/video_cache')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    store = ReviewStore(args.result, args.cache)
    store.read()
    store.prepare_videos()
    print(f'F1 blind review: http://127.0.0.1:{args.port}\nCSV: {store.csv_path}', flush=True)
    with ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(store)) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
