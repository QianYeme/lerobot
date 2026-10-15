"""F1 r1 read-only signal preflight; never fits probes without frozen independent labels."""
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

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/force/f1_formal4_signal_feasibility/index1'
DATA = ROOT / '数据集/formal4/kind_merged_nomaster'
DECISION = BASE / 'decision_F1_FORMAL4_CONTACT_SIGNAL_FEASIBILITY_r1.md'
SPLIT = ROOT / 'experiments/data/d1_training_readiness/index1/result/split_manifest_v1.json'


def features(state, action, width):
    """At t use observations <=t and sent commands <=t-1, never cross episodes."""
    assert width in (1, 9, 15)
    pos = np.asarray(state[:, 5], dtype=float)
    cmd = np.full(len(pos), np.nan)
    cmd[1:] = action[:-1, 5]
    dpos = np.r_[np.nan, np.diff(pos)]
    dcmd = np.r_[np.nan, np.diff(cmd)]
    channels = np.column_stack([pos, dpos, cmd, dcmd, state[:, 6:8]])
    indices = np.arange(width + 1, len(pos))
    values = []
    for t in indices:
        window = channels[t - width + 1:t + 1]
        values.append(np.column_stack([window[-1], window.mean(0), window.std(0), window[-1] - window[0]]).ravel())
    matrix = np.asarray(values).reshape(-1, 24)
    # Common mask across all four probes; current-degenerate comparisons are marked separately.
    valid = np.isfinite(matrix).all(1)
    return indices, matrix, valid


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statistics(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    unique = np.unique(finite)
    return {'n': len(values), 'nonfinite': int((~np.isfinite(values)).sum()),
            'min': float(finite.min()), 'max': float(finite.max()),
            'mean': float(finite.mean()), 'std': float(finite.std()),
            'quantiles_0_1_5_25_50_75_95_99_100': np.quantile(finite, [0, .01, .05, .25, .5, .75, .95, .99, 1]).tolist(),
            'zero_rate': float((finite == 0).mean()), 'unique_count': len(unique),
            'integer_valued': bool(np.allclose(finite, np.round(finite))),
            'minimum_observed_quantization_step': float(np.diff(unique).min()) if len(unique) > 1 else None,
            'hardware_saturation_limits': 'UNKNOWN', 'hardware_saturation_hit_rate': None}


def main():
    # Stop rather than retroactively declare r1 preregistered over an earlier experiment.
    out = BASE / 'result'
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Existing F1 result: preserve it and inspect provenance before resuming; no overwrite.')
    prior_raw = ROOT / 'results/experiments/F1_FORMAL4_CONTACT_SIGNAL_FEASIBILITY'
    if prior_raw.exists() and any(prior_raw.iterdir()):
        raise RuntimeError('Prior F1 raw exists: inspect possible r0 results before any new comparison.')
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    raw = prior_raw / run
    raw.mkdir(parents=True)
    out.mkdir(parents=True, exist_ok=True)

    def save(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    split = json.loads(SPLIT.read_text(encoding='utf-8'))
    groups = {k: [e['episode_index'] for e in split['episodes'] if e['split'] == k]
              for k in ('train', 'dev', 'final_test')}
    assert [len(groups[k]) for k in groups] == [36, 12, 12]
    assert len(set(sum(groups.values(), []))) == 60
    allowed = groups['train'] + groups['dev']
    info = json.loads((DATA / 'meta/info.json').read_text(encoding='utf-8'))
    assert info['features']['observation.state']['names'][5:] == ['gripper.pos', 'gripper.load', 'gripper.curr']
    assert info['features']['action']['names'][5] == 'gripper.pos'
    assert info['fps'] == 30
    # Explicit row filter on the shared file; do not expose Final episode rows or statistics.
    frame = pd.read_parquet(DATA / 'data/chunk-000/file-000.parquet',
                            filters=[('episode_index', 'in', allowed)],
                            columns=['episode_index', 'frame_index', 'timestamp', 'observation.state', 'action'])
    assert set(frame.episode_index) == set(allowed)
    episode_stats, window_counts = [], []
    for ep in allowed:
        z = frame[frame.episode_index == ep].sort_values('frame_index')
        state, action = np.stack(z['observation.state']), np.stack(z['action'])
        assert state.shape == (len(z), 8) and action.shape == (len(z), 6)
        assert np.array_equal(z.frame_index, np.arange(len(z)))
        error = float(np.max(np.abs(z.timestamp.to_numpy() - np.arange(len(z)) / info['fps'])))
        assert error < 1e-4
        group = 'train' if ep in groups['train'] else 'dev'
        signals = {name: statistics(state[:, i]) for name, i in [('pos', 5), ('load', 6), ('current', 7)]}
        episode_stats.append({'episode': ep, 'split': group, 'frames': len(z), 'timestamp_max_error_s': error,
                              'signals': signals,
                              'drift_first_last_30_mean_difference': {name: float(state[-30:, i].mean() - state[:30, i].mean()) for name, i in [('pos', 5), ('load', 6), ('current', 7)]},
                              'drift_caveat': 'within-episode task change, not proof of sensor drift'})
        for width in (1, 9, 15):
            indices, matrix, valid = features(state, action, width)
            np.savez_compressed(raw / f'features_ep{ep:03d}_w{width:02d}.npz', frames=indices, features=matrix, common_valid=valid)
            window_counts.append({'episode': ep, 'split': group, 'window': width, 'valid': int(valid.sum()),
                                  'insufficient_history': min(width + 1, len(z)), 'nonfinite_excluded': int((~valid).sum())})
    aggregate = {}
    for group in ('train', 'dev'):
        z = frame[frame.episode_index.isin(groups[group])]
        state = np.stack(z['observation.state'])
        aggregate[group] = {name: statistics(state[:, i]) for name, i in [('pos', 5), ('load', 6), ('current', 7)]}
    save('signal_integrity.json', {'schema': info['features'], 'fps': info['fps'], 'by_split': aggregate,
         'per_episode': episode_stats, 'current_constant_on_train': aggregate['train']['current']['unique_count'] == 1,
         'calibration': 'NOT_CALIBRATED', 'limits': 'sample extrema are NOT hardware saturation thresholds'})
    config = {'windows': [1, 9, 15], 'primary_window': 15, 'C': 1, 'penalty': 'l2',
              'class_weight': 'balanced', 'random_state': 20261014, 'solver': 'lbfgs', 'max_iter': 2000, 'tol': 1e-4,
              'template': ['current_available', 'mean', 'std_ddof0', 'last_minus_first'],
              'channel_order': ['pos', 'dpos', 'command_t_minus_1', 'dcommand_t_minus_1', 'load', 'current'],
              'probe_columns': {'S0': list(range(16)), 'S1': list(range(20)),
                                'S_current': list(range(16)) + list(range(20, 24)), 'S2': list(range(24))},
              'class_order': 'PENDING_TRAIN_LABEL_COVERAGE_GATE', 'bootstrap_replicates': 2000,
              'bootstrap_seed': 20261014, 'delta': .03, 'scaler': 'Train only; NOT FITTED', 'fitted': False}
    save('probe_config.json', config)
    save('leakage_audit.json', {'final_contents_read': False, 'episode_scoped_history': True,
         'max_command_index_at_t': 't-1', 'frame_time_episode_as_features': False,
         'train_only_scaler': 'NOT_FITTED', 'labeler_signal_access': 'blind_review folder contains no load/current or probe scores',
         'common_frame_mask': 'same 24-column finite mask for all probes',
         'cross_window_common_frames': 'Use intersection of valid frame IDs per episode; evaluation not run',
         'window_counts': window_counts})
    sources = [DECISION, SPLIT, DATA / 'meta/info.json', Path(__file__),
               ROOT / 'src/lerobot/robots/so_follower/so_follower.py',
               ROOT / 'src/lerobot/scripts/lerobot_record.py',
               ROOT / 'src/lerobot/motors/feetech/feetech.py', ROOT / 'src/lerobot/motors/feetech/tables.py',
               ROOT / 'src/lerobot/motors/motors_bus.py']
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'pyarrow', 'scikit-learn', 'opencv-python', 'matplotlib')}
    provenance = {'run_id': run, 'python': platform.python_version(), 'dependencies': versions,
                  'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
                  'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                  'sha256': {str(p.relative_to(ROOT)): digest(p) for p in sources}, 'groups': groups,
                  'r0_check': 'No pre-existing F1 result/raw found at start; no prior F1 scientific metrics found',
                  'actual_dataset': str(DATA), 'dataset_revision_manifest': split['dataset_revision'],
                  'historical_collection_source_identity': 'UNVERIFIED',
                  'raw': str(raw), 'cli': 'python scripts/audit_f1_signals.py'}
    save('provenance.json', provenance)
    (raw / 'start_dirty.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary'], cwd=ROOT))
    shutil.copy2(__file__, raw / Path(__file__).name)
    shutil.copy2(DECISION, raw / DECISION.name)
    save('task_freeze.json', {'status': 'NOT_FROZEN', 'reason': 'First six Train blind continuous-video review not executed; must not choose contact/phase using signal values or Dev', 'train_first_six_manifest_order': groups['train'][:6]})
    blind = out / 'evidence_samples'
    blind.mkdir()
    dense_path = ROOT / 'results/experiments/D2_1_ACTION_CHAIN_EVIDENCE_COMPLETION/dense_review/manifest.json'
    dense = {x['episode']: x for x in json.loads(dense_path.read_text())}
    blind_records = []
    for ep in groups['train'][:6]:
        source = Path(dense[ep]['clip'])
        dest = blind / f'train_ep{ep:03d}_dual_view.mp4'
        shutil.copy2(source, dest)
        cap = cv2.VideoCapture(str(dest))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert count == dense[ep]['decoded_frames_per_camera']
        for f in (100, 150, 220, 270, 380):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, image = cap.read()
            assert ok
            cv2.putText(image, f'Train ep{ep} frame{f} {f/30:.3f}s', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
            assert cv2.imwrite(str(blind / f'ep{ep:03d}_f{f:03d}.jpg'), image)
        cap.release()
        blind_records.append({'episode': ep, 'split': 'train', 'clip': str(dest.relative_to(out)),
                              'frames': count, 'fps': 30, 'review_status': 'PREPARED_NOT_REVIEWED_FOR_F1',
                              'source_sha256': digest(source), 'copied_sha256': digest(dest),
                              'original_video_offset_frames': dense[ep]['offset_frame']})
    save('label_manifest.json', {'label_source': 'PENDING_INDEPENDENT_RGB_REVIEW', 'records': blind_records,
         'historical_D2_events_are_not_F1_labels': True, 'reviewer_must_not_access': ['signal_integrity.json', 'figures/', str(raw)],
         'independent_dev_review': 'NOT_STARTED_BEFORE_TRAIN_TASK_FREEZE'})
    save('label_coverage.json', {'status': 'NOT_EVALUATED', 'reliable_target_frames': None,
         'reason': 'No independent F1 labels; not a statement of zero visible contact'})
    for name in ('metrics.json', 'bootstrap.json', 'confusion_matrices.json', 'contact_matching.json'):
        save(name, {'status': 'NOT_RUN', 'reason': 'Independent label/Train task-freeze gate pending', 'values': None})
    with (blind / 'review_intervals.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['episode', 'label', 'first_reliable_frame', 'last_reliable_frame', 'uncertain_start', 'uncertain_end', 'reviewer', 'evidence', 'reason'])
        for ep in groups['train'][:6]:
            writer.writerow([ep, '', '', '', '', '', '', f'train_ep{ep:03d}_dual_view.mp4', ''])
    (out / 'per_episode_metrics.csv').write_text('task,window,probe,episode,BA,macro_F1,per_class_recall,n_windows,status\n', encoding='utf-8')
    figure_dir = out / 'figures'
    figure_dir.mkdir()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, (name, i) in zip(axes, [('pos', 5), ('load', 6), ('current', 7)]):
        for group in ('train', 'dev'):
            values = np.stack(frame[frame.episode_index.isin(groups[group])]['observation.state'])[:, i]
            ax.hist(values[np.isfinite(values)], bins=40, density=True, alpha=.5, label=group)
        ax.set_title(name + ' (native units)')
        ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / 'signal_distributions.png', dpi=150)
    plt.close(fig)
    (figure_dir / 'README.md').write_text('Signal distributions only. Window/probe/delta/bootstrap plots NOT_APPLICABLE until independent labels and fitted probes exist. Do not show this folder to the blind label reviewer.\n', encoding='utf-8')
    validation = {'OVERALL_STATUS': 'PARTIAL_PASS', 'F1_SIGNAL_FEASIBILITY_COMPLETE': False,
                  'SCIENTIFIC_OUTCOME': 'UNRESOLVED', 'LOAD_PRIMARY_SUPPORTED': False,
                  'SUPPORT_SOURCE': 'NONE', 'NEXT_POLICY_TRAINING_AUTHORIZED': False,
                  'G0': 'PASS_WITH_HISTORICAL_COLLECTION_CODE_LIMITATION', 'G1': 'PARTIAL_SIGNAL_CHECKS_PASS_LABELS_PENDING',
                  'G2': 'FEATURES_IMPLEMENTED_TESTS_PENDING', 'G3': 'NOT_RUN_LABEL_GATE', 'G4': 'NOT_RUN',
                  'G5': 'NOT_RUN', 'G6': 'CURRENT_SOURCE_CAUSAL_TIMING_ONLY', 'G7': 'NA', 'G8': 'UNRESOLVED',
                  'continuous_video_playback_capability': False, 'window_counts': window_counts,
                  'no_robot_policy_training_no_deployment_change_no_final_no_push': True}
    save('validation.json', validation)
    (raw / 'audit_finished.json').write_text(json.dumps({'run_id': run, 'signal_preflight_complete': True, 'probes_fitted': False}), encoding='utf-8')
    print(json.dumps({'status': 'SIGNAL_PREFLIGHT_COMPLETE_LABEL_GATE_PENDING', 'train_frames': sum(x['frames'] for x in episode_stats if x['split'] == 'train'),
                      'dev_frames': sum(x['frames'] for x in episode_stats if x['split'] == 'dev'), 'first_six_train': groups['train'][:6], 'raw': str(raw)}))


if __name__ == '__main__':
    main()
