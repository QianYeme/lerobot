"""Prepare RGB-only review for the ten frozen Formal5 trials; never inspect force effects."""
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/force/f1rm_teleop_load_retrospective_match/index1/result'
PARENT = ROOT / 'experiments/force/f1r_actuator_load_contact_intervention/index1/result'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = OUT / 'rgb_review'
    evidence.mkdir(exist_ok=True)
    audit = json.loads((PARENT / 'collection_preflight.json').read_text(encoding='utf-8'))
    assert len(audit['trials']) == 10
    records = []
    hashes = {}
    for trial in audit['trials']:
        kind, episode = trial['dataset'], trial['episode']
        condition = {'formal5_kind1': 'FREE_CLOSE', 'formal5_kind2': 'CUP_CONTACT'}[kind]
        trial_id = f'{kind}_ep{episode:03d}'
        dataset = ROOT / '数据集/formal5' / kind
        # Read current original parquet, not merely the previous audit's cached arrays.
        data_path = dataset / 'data/chunk-000/file-000.parquet'
        frame = pd.read_parquet(data_path)
        frame = frame[frame.episode_index == episode].sort_values('frame_index')
        state, action = np.stack(frame['observation.state']), np.stack(frame.action)
        assert state.shape == (150, 9) and action.shape == (150, 6)
        assert np.isfinite(state).all() and np.isfinite(action).all()
        assert np.array_equal(frame.frame_index.to_numpy(), np.arange(150))
        assert np.allclose(np.diff(frame.timestamp), 1 / 30, atol=1e-6)
        hashes[str(data_path.relative_to(ROOT))] = digest(data_path)
        meta = pd.read_parquet(dataset / 'meta/episodes/chunk-000/file-000.parquet')
        row = meta[meta.episode_index == episode].iloc[0]
        views = []
        for key in ['observation.images.top', 'observation.images.gripper']:
            prefix = 'videos/' + key
            source = dataset / prefix / f"chunk-{int(row[prefix+'/chunk_index']):03d}" / f"file-{int(row[prefix+'/file_index']):03d}.mp4"
            hashes[str(source.relative_to(ROOT))] = digest(source)
            cap = cv2.VideoCapture(str(source))
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(float(row[prefix + '/from_timestamp']) * 30))
            images = []
            for _ in range(150):
                ok, image = cap.read()
                assert ok, (trial_id, key, len(images))
                images.append(image)
            cap.release()
            views.append(images)
        clip = evidence / f'{trial_id}.mp4'
        # Every frame in order; no one-frame-per-second replacement.
        command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                   '-s', '1280x480', '-r', '30', '-i', '-', '-an', '-c:v', 'libx264',
                   '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(clip)]
        proc = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for i in range(150):
                proc.stdin.write(np.hstack([views[0][i], views[1][i]]).tobytes())
        finally:
            proc.stdin.close()
        assert proc.wait() == 0
        check = cv2.VideoCapture(str(clip))
        assert int(check.get(cv2.CAP_PROP_FRAME_COUNT)) == 150
        assert check.get(cv2.CAP_PROP_FPS) == 30
        check.release()
        records.append(dict(trial_id=trial_id, condition=condition, episode=episode, frames=150,
                            nominal_fps=30, clip=f'rgb_review/{clip.name}',
                            dataset=str(dataset.relative_to(ROOT)), video_sha256=digest(clip)))
    with (OUT / 'trial_manifest.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    labels = OUT / 'contact_annotations.csv'
    if not labels.exists():
        labels.write_text('trial_id,label,first_reliable_frame,last_reliable_frame,reviewer,reason\n', encoding='utf-8-sig')
    write_json(OUT / 'input_validation.json', dict(trials=10, frames=1500, frames_per_trial=150,
               schema_finite_contiguous_timestamp_checks='PASS', original_file_sha256=hashes,
               RGB_frames_decoded=3000, clips='10 clips, 150 frames each, 30FPS, 1280x480',
               contact_labels='PENDING_INDEPENDENT_RGB_REVIEW', force_effects_inspected=False))
    config = dict(experiment_id='F1RM_TELEOP_LOAD_RETROSPECTIVE_MATCH', revision='r0',
                  primary_distance=0.75, sensitivity_distance=1.00,
                  sensitivity_only_if_primary_coverage_fails=True,
                  features=['pos[t]', 'pos[t]-pos[t-1]', 'command[t-1]', 'command[t-1]-command[t-2]'],
                  first_eligible_frame=2, scaler='population mean/std (ddof=0), zero SD -> scale 1',
                  scaler_fit='all annotated FREE_CLOSE closing + CUP_CONTACT PRECONTACT, excluding frame<2',
                  target_order='trial_id ascending, then frame ascending; fixed before any matching',
                  source_tie_order='trial_id ascending, then frame ascending',
                  matching='global FREE_CLOSE pool, Euclidean, greedy 1:1 without replacement per pass',
                  SMD='(mean_contact-mean_free)/sqrt((var_contact+var_free)/2), sample variance ddof=1',
                  zero_variance_SMD='0 for equal means; signed infinity otherwise; undefined if n<2',
                  bootstrap=dict(unit='CUP_CONTACT trial', resamples=2000, seed=20261016),
                  implementation_choices='Frozen before labels/matching/load effects; no optional features')
    config_path = OUT / 'matching_config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text(encoding='utf-8')) == config, 'Never overwrite a different freeze'
    else:
        write_json(config_path, config)
    template = (ROOT / 'tools/f1rm_review.html').read_text(encoding='utf-8')
    (OUT / 'review.html').write_text(template.replace('__TRIALS__', json.dumps(records)), encoding='utf-8')
    write_json(OUT / 'metrics.json', dict(OVERALL_STATUS='PARTIAL_PASS', F1RM_COMPLETE=False,
               SCIENTIFIC_OUTCOME='UNRESOLVED_PENDING_INDEPENDENT_CONTACT_LABELS',
               EVIDENCE_SCOPE='EXPLORATORY_ONLY', validated_trials=10, primary_matching_executed=False,
               force_effects_inspected=False))
    print(json.dumps(dict(result=str(OUT), trials=len(records), frames=1500, clips_verified=10)))


if __name__ == '__main__':
    main()
