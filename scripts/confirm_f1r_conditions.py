"""Record operator-confirmed F1R condition mapping; preserve preflight and prepare RGB previews."""
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/force/f1r_actuator_load_contact_intervention/index1/result'
MAPPING = {'formal5_kind1': 'FREE_CLOSE', 'formal5_kind2': 'CUP_CONTACT'}


def main():
    audit_path = OUT / 'collection_preflight.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    run = ROOT / 'results/experiments/F1R_ACTUATOR_LOAD_CONTACT_INTERVENTION' / ('conditions_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    run.mkdir(parents=True)
    shutil.copy2(audit_path, run / 'preflight_before_mapping.json')
    mapping = {'mapping': MAPPING, 'source': 'USER_EXPLICIT_CONFIRMATION',
               'statement': 'kind1＝空夹，kind2＝带杯', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
               'command_program_equivalence': 'STILL_UNVERIFIED', 'raw': str(run)}
    (OUT / 'condition_mapping.json').write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding='utf-8')
    for trial in audit['trials']:
        trial['condition'] = MAPPING[trial['dataset']]
    audit['condition_mapping'] = MAPPING
    audit['condition_mapping_source'] = 'USER_EXPLICIT_CONFIRMATION'
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    with (OUT / 'trial_manifest.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['trial_id', 'dataset', 'episode', 'condition', 'frames', 'nominal_fps', 'raw'])
        writer.writeheader()
        for t in audit['trials']:
            writer.writerow({'trial_id': f"{t['dataset']}_ep{t['episode']:03d}", **{k: t[k] for k in ['dataset', 'episode', 'condition', 'frames', 'nominal_fps', 'raw']}})
    evidence = OUT / 'evidence_samples'
    evidence.mkdir(exist_ok=True)
    for kind, condition in MAPPING.items():
        dataset = ROOT / '数据集/formal5' / kind
        meta = pd.read_parquet(dataset / 'meta/episodes/chunk-000/file-000.parquet')
        r = meta[meta.episode_index == 0].iloc[0]
        views = []
        for key in ['observation.images.top', 'observation.images.gripper']:
            prefix = 'videos/' + key
            source = dataset / 'videos' / key / f"chunk-{int(r[prefix+'/chunk_index']):03d}" / f"file-{int(r[prefix+'/file_index']):03d}.mp4"
            cap = cv2.VideoCapture(str(source))
            offset = round(float(r[prefix + '/from_timestamp']) * 30)
            images = []
            for frame in [0, 60, 100, 149]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, offset + frame)
                ok, image = cap.read()
                assert ok
                images.append(cv2.resize(image, (384, 288)))
            cap.release()
            views.append(images)
        rows = []
        for i, frame in enumerate([0, 60, 100, 149]):
            image = np.hstack([views[0][i], views[1][i]])
            cv2.putText(image, f'{kind} ep0 f{frame} {condition}', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
            rows.append(image)
        assert cv2.imwrite(str(evidence / f'{kind}_ep000_rgb_preview.jpg'), np.vstack(rows))
    shutil.copy2(__file__, run / Path(__file__).name)
    print(json.dumps(mapping, ensure_ascii=False))


if __name__ == '__main__':
    main()
