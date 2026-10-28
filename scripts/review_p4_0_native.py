"""Decode each native video once and generate episode review sheets; no QC inference."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def main():
    root = Path('outputs/p4_0/formal6_merged')
    out = Path('outputs/p4_0/video_review')
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(root / 'meta/episodes/chunk-000/file-000.parquet')
    records = []
    for camera in ('top', 'gripper'):
        key = 'observation.images.' + camera
        for file_id, group in meta.groupby('videos/' + key + '/file_index'):
            chunks = group['videos/' + key + '/chunk_index'].unique()
            assert len(chunks) == 1
            path = root / 'videos' / key / f'chunk-{int(chunks[0]):03d}' / f'file-{int(file_id):03d}.mp4'
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            assert abs(fps - 30) < 1e-5
            wanted, sheets = {}, {}
            expected = 0
            for row in group.to_dict('records'):
                ep, length = int(row['episode_index']), int(row['length'])
                start = round(row['videos/' + key + '/from_timestamp'] * fps)
                assert abs(start / fps - row['videos/' + key + '/from_timestamp']) < 1e-4
                expected = max(expected, start + length)
                sheets[ep] = Image.new('RGB', (1280, 840), '#161616')
                for n, frame in enumerate(np.linspace(0, length - 1, 16).astype(int)):
                    wanted.setdefault(start + int(frame), []).append((ep, int(frame), n))
            count, digest = 0, hashlib.sha256()
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                assert bgr.shape == (480, 640, 3)
                if count in wanted:
                    for ep, frame, n in wanted[count]:
                        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                        image.thumbnail((320, 180))
                        sheet = sheets[ep]
                        x, y = n % 4 * 320, n // 4 * 210
                        sheet.paste(image, (x, y + 25))
                        ImageDraw.Draw(sheet).text((x + 3, y + 5), f'ep{ep:03d} {camera} f{frame} {frame/fps:.2f}s', fill='white')
                        digest.update(bgr.tobytes())
                count += 1
            cap.release()
            assert count >= expected, (path, count, expected)
            for ep, sheet in sheets.items():
                sheet.save(out / f'ep{ep:03d}_{camera}.jpg')
            records.append(dict(path=str(path), decoded_frames=count, required_frames=expected, fps=fps, sampled_pixel_sha256=digest.hexdigest()))
            (out / 'decode_progress.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
            print(camera, int(file_id), count, 'DECODED', flush=True)
    (out / 'decode_audit.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print('DECODE_COMPLETE_NOT_VISUAL_QC', flush=True)


if __name__ == '__main__':
    main()
