"""Freeze the new F1P subset before preparing its six independent blind RGB clips."""
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from run_f1_review_ui import DEFAULT_RESULT, FIELDS, ReviewStore

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/force/f1p_contact_signal_pilot/index1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = BASE / 'result'
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Existing F1P evidence found; no overwrite. Inspect before resuming.')
    split_path = ROOT / 'experiments/data/d1_training_readiness/index1/result/split_manifest_v1.json'
    split = json.loads(split_path.read_text())
    train = [x['episode_index'] for x in split['episodes'] if x['split'] == 'train']
    fit = [0, 2, 4, 6, 7, 9]
    assert len(train) == 36 and set(fit) <= set(train)
    remaining = [ep for ep in train if ep not in fit]
    assert len(remaining) == 30
    evaluate = [remaining[i] for i in [0, 5, 11, 17, 23, 29]]
    assert len(set(evaluate)) == 6 and not set(fit) & set(evaluate)
    old = ReviewStore(DEFAULT_RESULT, ROOT / 'outputs/f1_rgb_review_ui/video_cache')
    fit_rows = old.read()
    for ep in fit:
        selected = [r for r in fit_rows if int(r['episode']) == ep and r['label']]
        old.validate(ep, selected)
        assert {'PRECONTACT', 'CONTACT_SUPPORTED'} <= {r['label'] for r in selected}
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    raw = ROOT / 'results/experiments/F1P_CONTACT_SIGNAL_PILOT' / run_id
    raw.mkdir(parents=True)
    out.mkdir(parents=True)

    def save(name, obj):
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

    contract = BASE / 'decision_F1P_CONTACT_SIGNAL_PILOT.md'
    subset = {'experiment_id': 'F1P_CONTACT_SIGNAL_PILOT', 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'source_manifest': str(split_path.relative_to(ROOT)), 'source_manifest_sha256': sha(split_path),
              'source_train_order': train, 'pilot_fit': fit, 'remaining_train_order': remaining,
              'selection_zero_based_indices': [0, 5, 11, 17, 23, 29], 'pilot_eval': evaluate,
              'selection_code': 'remaining=[ep for ep in Train_manifest_order if ep not in [0,2,4,6,7,9]]; eval=[remaining[i] for i in [0,5,11,17,23,29]]',
              'groups_disjoint': True, 'both_groups_original_train': True,
              'official_dev_and_final_contents_read_this_run': False, 'fit_label_csv_sha256': sha(old.csv_path),
              'contract_sha256': sha(contract), 'scores_seen_before_freeze': False}
    # Freeze FIRST, before opening selected data, new RGB evidence, or fitting anything.
    save('subset_manifest.json', subset)
    shutil.copy2(out / 'subset_manifest.json', raw / 'subset_manifest_frozen.json')
    shutil.copy2(old.csv_path, raw / 'pilot_fit_original_all_labels.csv')
    shutil.copy2(contract, raw / contract.name)
    shutil.copy2(__file__, raw / Path(__file__).name)
    sources = [Path(__file__), ROOT / 'scripts/run_f1_review_ui.py', ROOT / 'scripts/audit_f1_signals.py',
               ROOT / 'src/lerobot/scripts/lerobot_record.py', ROOT / 'src/lerobot/robots/so_follower/so_follower.py']
    save('provenance.json', {'run_id': run_id, 'raw': str(raw), 'python': platform.python_version(),
         'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'pyarrow', 'scikit-learn', 'opencv-python')},
         'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
         'status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
         'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sources},
         'reused_audits': ['experiments/force/f1_formal4_signal_feasibility/index1/result/signal_timing_audit.md',
                           'tests/test_f1_causal_features.py'], 'old_audits_recomputed': False,
         'cli': 'python scripts/prepare_f1p_pilot.py'})
    (raw / 'start_dirty.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary'], cwd=ROOT))
    data = ROOT / '数据集/formal4/kind_merged_nomaster'
    frame = pd.read_parquet(data / 'data/chunk-000/file-000.parquet',
                            filters=[('episode_index', 'in', fit + evaluate)],
                            columns=['episode_index', 'frame_index', 'timestamp', 'observation.state', 'action'])
    assert set(frame.episode_index) == set(fit + evaluate)
    info = json.loads((data / 'meta/info.json').read_text())
    assert info['fps'] == 30
    samples = out / 'evidence_samples'
    samples.mkdir()
    with (samples / 'pilot_fit_labels_snapshot.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(fit_rows)  # Preserve prior phase labels too; only two classes used in this pilot.
    records, fit_counts = [], []
    for ep in fit + evaluate:
        z = frame[frame.episode_index == ep].sort_values('frame_index')
        assert np.array_equal(z.frame_index, np.arange(len(z)))
        state, action = np.stack(z['observation.state']), np.stack(z.action)
        assert state.shape == (len(z), 8) and action.shape == (len(z), 6)
        assert np.isfinite(state).all() and np.isfinite(action).all()
        assert np.max(np.abs(z.timestamp.to_numpy() - np.arange(len(z)) / 30)) < 1e-4
        np.savez_compressed(raw / f'episode_{ep:03d}.npz', frames=z.frame_index.to_numpy(), state=state, action=action)
        if ep in fit:
            counts = {}
            for label in ('PRECONTACT', 'CONTACT_SUPPORTED'):
                ids = set()
                for r in fit_rows:
                    if int(r['episode']) == ep and r['label'] == label:
                        ids.update(range(int(r['first_reliable_frame']), int(r['last_reliable_frame']) + 1))
                counts[label] = len([f for f in ids if f >= 16])
            fit_counts.append({'episode': ep, 'reliable_target_frames_W15': counts,
                               'evaluable': all(n >= 5 for n in counts.values())})
            continue
        source = ROOT / f'results/experiments/D2_1_ACTION_CHAIN_EVIDENCE_COMPLETION/dense_review/ep{ep:03d}/dual_view.mp4'
        dest = samples / f'pilot_eval_train_ep{ep:03d}_dual_view.mp4'
        shutil.copy2(source, dest)
        cap = cv2.VideoCapture(str(dest))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(z)
        assert abs(cap.get(cv2.CAP_PROP_FPS) - 30) < .01
        cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        ok, image = cap.read()
        cap.release()
        assert ok and cv2.imwrite(str(samples / f'train_ep{ep:03d}_frame_100.jpg'), image)
        records.append({'episode': ep, 'split': 'train', 'pilot_role': 'pilot_eval',
                        'frames': len(z), 'fps': 30, 'clip': str(dest.relative_to(out)),
                        'source_sha256': sha(source), 'copied_sha256': sha(dest),
                        'review_status': 'PREPARED_NOT_REVIEWED'})
    save('label_manifest.json', {'experiment_id': 'F1P_CONTACT_SIGNAL_PILOT', 'records': records,
         'pilot_fit_labels': 'evidence_samples/pilot_fit_labels_snapshot.csv', 'fit_label_sha256': sha(old.csv_path),
         'fit_source': 'HUMAN_REVIEWED_BY_lyj', 'new_labels': 'PENDING_INDEPENDENT_HUMAN_RGB_REVIEW',
         'codex_independent_continuous_playback': False})
    with (samples / 'review_intervals.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([{'episode': ep} for ep in evaluate])
    save('label_coverage.json', {'pilot_fit': fit_counts, 'pilot_fit_qualifying': sum(c['evaluable'] for c in fit_counts),
         'pilot_eval': [{'episode': ep, 'status': 'NOT_REVIEWED', 'reliable_frames': None} for ep in evaluate],
         'pilot_eval_qualifying': None, 'required_each_group': 4, 'fit_probe_permitted_now': False})
    config = json.loads((DEFAULT_RESULT / 'probe_config.json').read_text())
    config.update({'random_state': 20261015, 'bootstrap_seed': 20261015, 'scaler': 'PILOT_FIT_ONLY_NOT_FITTED',
                   'class_order': ['PRECONTACT', 'CONTACT_SUPPORTED'], 'maximum_fits': 12, 'fits_performed': 0})
    save('probe_config.json', config)
    save('leakage_audit.json', {'subset_frozen_before_new_labels_and_any_score': True,
         'new_source_data_episodes': fit + evaluate, 'official_dev_or_final_read': False,
         'scaler_fit': False, 'features': 'reuse audited causal template; obs<=t, commands<=t-1',
         'absolute_time_frame_episode_inputs': False, 'blind_UI': 'RGB and interval annotations only',
         'historical_collection_code_firmware_identity': 'UNVERIFIED; inherited limitation retained'})
    for name in ('metrics.json', 'bootstrap.json'):
        save(name, {'status': 'NOT_RUN', 'reason': 'Pilot-eval six human RGB labels pending', 'scope': 'EXPLORATORY_ONLY'})
    save('validation.json', {'OVERALL_STATUS': 'PARTIAL_PASS', 'PILOT_COMPLETE': False, 'PILOT_OUTCOME': 'UNRESOLVED',
         'EVIDENCE_SCOPE': 'EXPLORATORY_ONLY', 'ORIGINAL_F1_COMPLETE': False, 'NEXT_POLICY_TRAINING_AUTHORIZED': False,
         'subset_selection_and_disjointness': 'PASS', 'selected_data_schema_finite_frame_timing': 'PASS',
         'fit_labels_schema_and_coverage': 'PASS', 'eval_labels': 'PENDING', 'probe_fits': 0,
         'policy_training': False, 'dev_final_contents_read_this_run': False})
    (out / 'per_episode_metrics.csv').write_text('role,episode,window,probe,BA,macro_F1,per_class_recall,n_windows,status\n', encoding='utf-8')
    (out / 'figures').mkdir()
    (out / 'figures/README.md').write_text('NOT_RUN: comparisons, episode deltas and bootstrap plots require frozen new labels and fitted probes. No placeholder numeric plots.\n')
    (out / 'confound_review.md').write_text('# 混杂审查尚未执行\n\n独立验证标签尚未完成，未查看分类器成绩。后续须检查S0饱和、接触前闭爪/接触后的位置和历史运动重叠及预测错误；图示不能排除因果混杂。\n', encoding='utf-8')
    (out / 'proposal.md').write_text('# 当前下一步\n\n仅在独立页面审核冻结的六个pilot-eval原Train片段，每集两类可靠区间；无需重标原六集或追加阶段类。标签覆盖通过后执行本合同最多12次固定probe拟合。无policy训练权限。\n', encoding='utf-8')
    print(json.dumps({'pilot_fit': fit, 'pilot_eval': evaluate, 'subset_sha256': sha(out / 'subset_manifest.json'), 'raw': str(raw)}))


if __name__ == '__main__':
    main()
