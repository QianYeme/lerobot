"""Produce full-resolution evidence and exact review clips from permitted episodes."""
import json
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/position/d2_1_evidence_completion/index1/result'
RAW = ROOT / 'results/experiments/D2_1_ACTION_CHAIN_EVIDENCE_COMPLETION/dense_review'
manifest = json.loads((RAW / 'manifest.json').read_text())
evidence = []
for item in manifest:
    ep = item['episode']
    folder = OUT / 'manual_review' / f'ep{ep:03d}'
    folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(item['clip'])
    windows = [('clamp_lift_hold_return', 120, 359), ('contact_release', 330, item['decoded_frames_per_camera'] - 1)]
    writers = {name: cv2.VideoWriter(str(folder / f'{name}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 480)) for name, _, _ in windows}
    stills = []
    for frame in range(item['decoded_frames_per_camera']):
        ok, image = cap.read()
        assert ok, (ep, frame)
        cv2.putText(image, f'ep {ep:03d} frame {frame:03d} {frame / 30:.3f}s', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
        for name, start, end in windows:
            if start <= frame <= end:
                writers[name].write(image)
        if frame in [120, 180, 220, 250, 280, 310, 340, 380, 420, 450]:
            name = f'frame_{frame:03d}.jpg'
            assert cv2.imwrite(str(folder / name), image)
            stills.append(name)
    cap.release()
    for writer in writers.values():
        writer.release()
    evidence.append({'episode': ep, 'split': item['split'], 'source_clip': item['clip'],
                     'clips': [{'path': str(folder / f'{name}.mp4'), 'frame_start': start, 'frame_end': end, 'time_start_s': start / 30, 'time_end_s': end / 30} for name, start, end in windows],
                     'full_resolution_stills': [str(folder / name) for name in stills]})
    print(item['split'], ep, flush=True)
(OUT / 'manual_evidence_manifest.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
