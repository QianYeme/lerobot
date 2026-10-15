"""Run the frozen 6-fit/6-eval F1P linear probes once, with immutable label snapshots."""
import argparse
import csv
import hashlib
import json
import math
import pickle
import shutil
import subprocess
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

if __package__:
    from .audit_f1_signals import features
    from .run_f1_review_ui import DEFAULT_RESULT, ReviewStore
else:
    from audit_f1_signals import features
    from run_f1_review_ui import DEFAULT_RESULT, ReviewStore

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/force/f1p_contact_signal_pilot/index1/result'
CLASSES = ['PRECONTACT', 'CONTACT_SUPPORTED']
COLS = {'S0': list(range(16)), 'S1': list(range(20)),
        'S_current': list(range(16)) + list(range(20, 24)), 'S2': list(range(24))}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def expand_labels(rows, episode, length):
    """Unknown intervals exclude target frames; never infer labels outside annotated spans."""
    labels = np.full(length, -1, dtype=int)
    unknown = np.zeros(length, dtype=bool)
    for r in rows:
        if int(r['episode']) != episode:
            continue
        if r['uncertain_start']:
            a, b = int(r['uncertain_start']), int(r['uncertain_end'])
            unknown[a:b + 1] = True
        if r['label'] in CLASSES:
            a, b = int(r['first_reliable_frame']), int(r['last_reliable_frame'])
            if not 0 <= a <= b < length:
                raise ValueError('Label outside episode')
            value = CLASSES.index(r['label'])
            if np.any((labels[a:b + 1] >= 0) & (labels[a:b + 1] != value)):
                raise ValueError('Conflicting contact labels')
            labels[a:b + 1] = value
    labels[unknown] = -1
    return labels, unknown


def paired_bootstrap(delta, indices):
    delta = np.asarray(delta, dtype=float)
    values = delta[indices].mean(axis=1)
    return {'point_estimate': float(delta.mean()), 'bootstrap_mean': float(values.mean()),
            'percentile_95_ci': np.quantile(values, [.025, .975]).tolist(),
            'positive_episodes': int((delta > 0).sum()), 'n_episodes': len(delta),
            'positive_fraction': float((delta > 0).mean())}


def score(y, pred):
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    support = matrix.sum(axis=1)
    recall = np.divide(matrix.diagonal(), support, out=np.zeros(2), where=support > 0)
    return {'BA': float(recall.mean()) if (support > 0).all() else None,
            'macro_F1': float(f1_score(y, pred, labels=[0, 1], average='macro', zero_division=0)),
            'per_class_recall': recall.tolist(), 'confusion_matrix': matrix.tolist(),
            'class_counts': support.tolist(), 'n_windows': len(y),
            'evaluable': bool((support >= 5).all())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--remote-output', type=Path)
    args = parser.parse_args()
    remote = args.remote_output
    config = json.loads((OUT / 'probe_config.json').read_text())
    if config.get('fits_performed', 0) or (OUT / 'label_freeze.json').exists():
        if remote is None:
            raise RuntimeError('F1P fitting already started/frozen. Preserve evidence; no automatic rerun/search.')
        previous = json.loads((OUT / 'label_freeze.json').read_text())
        previous_raw = Path(previous['raw'])
        for artifact in list(previous_raw.glob('probe_*.pkl')) + list(previous_raw.glob('pred_*.npz')):
            assert (remote / artifact.name).exists() and sha(artifact) == sha(remote / artifact.name)
        assert json.loads((OUT / 'metrics.json').read_text())['status'] == 'NOT_RUN'
        assert sha(OUT / 'evidence_samples/review_intervals.csv') == previous['eval_csv_sha256']
        assert sha(OUT / 'evidence_samples/pilot_fit_labels_snapshot.csv') == previous['fit_csv_sha256']
    subset_path = OUT / 'subset_manifest.json'
    subset = json.loads(subset_path.read_text())
    prep = json.loads((OUT / 'provenance.json').read_text())
    prep_raw = Path(prep['raw'])
    assert sha(subset_path) == sha(prep_raw / 'subset_manifest_frozen.json')
    assert sha(prep_raw / 'pilot_fit_original_all_labels.csv') == subset['fit_label_csv_sha256']
    fit_csv = OUT / 'evidence_samples/pilot_fit_labels_snapshot.csv'
    eval_csv = OUT / 'evidence_samples/review_intervals.csv'
    fit_rows, eval_rows = read_rows(fit_csv), read_rows(eval_csv)
    assert {int(r['episode']) for r in fit_rows} == set(subset['pilot_fit'])
    assert {int(r['episode']) for r in eval_rows} == set(subset['pilot_eval'])
    assert fit_rows == read_rows(prep_raw / 'pilot_fit_original_all_labels.csv')
    fit_store = ReviewStore(DEFAULT_RESULT, ROOT / 'outputs/f1_rgb_review_ui/video_cache')
    eval_store = ReviewStore(OUT, ROOT / 'outputs/f1_rgb_review_ui/video_cache')
    for ep in subset['pilot_fit']:
        fit_store.validate(ep, [r for r in fit_rows if int(r['episode']) == ep and r['label']])
    for ep in subset['pilot_eval']:
        eval_store.validate(ep, [r for r in eval_rows if int(r['episode']) == ep and r['label']])
    run = ROOT / 'results/experiments/F1P_CONTACT_SIGNAL_PILOT' / ('eval_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    run.mkdir(parents=True)
    for p in OUT.iterdir():
        if p.is_file():
            shutil.copy2(p, run / ('before_' + p.name))
    shutil.copy2(__file__, run / Path(__file__).name)
    shutil.copy2(fit_csv, run / 'fit_labels_frozen.csv')
    shutil.copy2(eval_csv, run / 'eval_labels_frozen.csv')
    (run / 'start_dirty.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary'], cwd=ROOT))

    def save(name, value):
        (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    data, coverage = {}, []
    for role, ids, rows in [('fit', subset['pilot_fit'], fit_rows), ('eval', subset['pilot_eval'], eval_rows)]:
        for ep in ids:
            with np.load(prep_raw / f'episode_{ep:03d}.npz') as z:
                state, action, frames = z['state'], z['action'], z['frames']
            assert np.array_equal(frames, np.arange(len(frames)))
            y, unknown = expand_labels(rows, ep, len(state))
            data[ep] = {'role': role, 'state': state, 'y_all': y, 'windows': {}}
            counts = {}
            for width in (1, 9, 15):
                ix, x, finite = features(state, action, width)
                valid = finite & (y[ix] >= 0)
                data[ep]['windows'][width] = {'frames': ix[valid], 'x': x[valid], 'y': y[ix[valid]]}
                counts[width] = {'class_counts': np.bincount(y[ix[valid]], minlength=2).tolist(),
                                 'history_insufficient': min(width + 1, len(state)),
                                 'nonfinite_windows_excluded': int((~finite).sum()),
                                 'unknown_or_unlabeled_targets_excluded': int((finite & (y[ix] < 0)).sum())}
                np.savez_compressed(run / f'features_ep{ep:03d}_w{width}.npz', frames=ix[valid], x=x[valid], y=y[ix[valid]])
            coverage.append({'episode': ep, 'role': role, 'windows': counts,
                             'unknown_frames': int(unknown.sum()), 'unlabeled_frames': int((y < 0).sum()),
                             'evaluable_W15': all(n >= 5 for n in counts[15]['class_counts'])})
    qualifying = {role: sum(r['role'] == role and r['evaluable_W15'] for r in coverage) for role in ('fit', 'eval')}
    freeze = {'frozen_before_fitting': True, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'fit_csv_sha256': sha(fit_csv), 'eval_csv_sha256': sha(eval_csv),
              'subset_sha256': sha(subset_path), 'class_order': CLASSES, 'raw': str(run),
              'review_source': 'HUMAN_REVIEWED_BY_lyj; user confirmed complete', 'auto_interpolated_labels': False}
    save('label_freeze.json', freeze)
    label_manifest = json.loads((OUT / 'label_manifest.json').read_text())
    label_manifest.update({'new_labels': 'HUMAN_REVIEWED_BY_lyj', 'label_freeze': freeze})
    for record in label_manifest['records']:
        record['review_status'] = 'HUMAN_REVIEWED_USER_CONFIRMED_COMPLETE'
    save('label_manifest.json', label_manifest)
    save('label_coverage.json', {'episodes': coverage, 'qualifying': qualifying, 'minimum_per_group': 4})
    if min(qualifying.values()) < 4:
        save('validation.json', {'OVERALL_STATUS': 'PASS', 'PILOT_COMPLETE': True,
             'PILOT_OUTCOME': 'UNRESOLVED', 'EVIDENCE_SCOPE': 'EXPLORATORY_ONLY',
             'ORIGINAL_F1_COMPLETE': False, 'NEXT_POLICY_TRAINING_AUTHORIZED': False,
             'stop_reason': 'Frozen coverage stop branch reached; no fitting', 'raw': str(run)})
        raise RuntimeError('Coverage stop branch reached; report explicit label gaps, do not fit.')
    # Fit at most twelve fixed classifiers. No search, resampling, or pilot-eval normalization.
    tasks = [(w, p) for w in (1, 9, 15) for p in COLS]
    remote_health = {}
    if remote:
        done = json.loads((remote / 'done.json').read_text())
        assert done['fits'] == 12 and done['fit_episodes'] == subset['pilot_fit'] and done['eval_episodes'] == subset['pilot_eval']
        remote_health = {(r['window'], r['probe']): r for r in json.loads((remote / 'health.json').read_text())}
        shutil.copy2(remote / 'done.json', run / 'remote_environment.json')
        save('execution_environment.json', {'probe_fitting': 'remote isolated CPU venv',
             'remote_versions_and_fit_ids': done, 'local_stage': 'feature/input verification, scoring, bootstrap and reports only',
             'local_failure': str(Path(json.loads((OUT / 'remote_execution.json').read_text())['local_failed_evidence']) / 'local_failure.json'), 'local_fits': 0})
    current_available = bool(np.ptp(np.concatenate([data[ep]['state'][:, 7] for ep in subset['pilot_fit']])) > 0)
    if not current_available:
        tasks = [(w, p) for w, p in tasks if p not in ('S_current', 'S2')]
    config.update({'fits_performed': len(tasks), 'fitted': False, 'scaler': 'PILOT_FIT_ONLY'})
    save('probe_config.json', config)  # Budget guard written before any fitting.

    def fit_one(task):
        width, probe = task
        cols = COLS[probe]
        xfit = np.concatenate([data[ep]['windows'][width]['x'][:, cols] for ep in subset['pilot_fit']])
        yfit = np.concatenate([data[ep]['windows'][width]['y'] for ep in subset['pilot_fit']])
        if remote:
            health_row = remote_health[task]
            assert health_row['converged'] and health_row['n_fit_windows'] == len(yfit)
            np.testing.assert_allclose(health_row['scaler_mean'], xfit.mean(axis=0), atol=1e-8)
            shutil.copy2(remote / f'probe_w{width}_{probe}.pkl', run / f'probe_w{width}_{probe}.pkl')
            outputs = []
            for role, ids in [('fit', subset['pilot_fit']), ('eval', subset['pilot_eval'])]:
                for ep in ids:
                    d = data[ep]['windows'][width]
                    path = remote / f'pred_ep{ep:03d}_w{width}_{probe}.npz'
                    with np.load(path) as z:
                        np.testing.assert_equal(z['frames'], d['frames'])
                        np.testing.assert_equal(z['y'], d['y'])
                        prediction, probability = z['pred'], z['probability']
                        assert np.isfinite(probability).all()
                        np.testing.assert_equal(prediction, (probability >= .5).astype(int))
                    shutil.copy2(path, run / path.name)
                    common = set.intersection(*[set(data[ep]['windows'][w]['frames']) for w in (1, 9, 15)])
                    for scope, mask in [('all_valid', np.ones(len(prediction), dtype=bool)),
                                        ('common_across_windows', np.isin(d['frames'], list(common)))]:
                        outputs.append({'role': role, 'episode': ep, 'window': width, 'probe': probe,
                                        'frame_scope': scope, **score(d['y'][mask], prediction[mask])})
            return outputs, health_row
        scaler = StandardScaler().fit(xfit)
        model = LogisticRegression(C=1, penalty='l2', class_weight='balanced', solver='lbfgs',
                                   max_iter=2000, tol=1e-4, random_state=20261015)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(scaler.transform(xfit), yfit)
        assert np.isfinite(model.coef_).all() and int(model.n_iter_.max()) < 2000
        with (run / f'probe_w{width}_{probe}.pkl').open('wb') as stream:
            pickle.dump({'scaler': scaler, 'model': model, 'columns': cols, 'fit_episodes': subset['pilot_fit']}, stream)
        outputs = []
        for role, ids in [('fit', subset['pilot_fit']), ('eval', subset['pilot_eval'])]:
            for ep in ids:
                d = data[ep]['windows'][width]
                probability = model.predict_proba(scaler.transform(d['x'][:, cols]))[:, 1]
                assert np.isfinite(probability).all()
                prediction = (probability >= .5).astype(int)
                np.savez_compressed(run / f'pred_ep{ep:03d}_w{width}_{probe}.npz',
                                    frames=d['frames'], y=d['y'], pred=prediction, probability=probability)
                outputs.append({'role': role, 'episode': ep, 'window': width, 'probe': probe,
                                'frame_scope': 'all_valid', **score(d['y'], prediction)})
                common = set.intersection(*[set(data[ep]['windows'][w]['frames']) for w in (1, 9, 15)])
                mask = np.isin(d['frames'], list(common))
                outputs.append({'role': role, 'episode': ep, 'window': width, 'probe': probe,
                                'frame_scope': 'common_across_windows', **score(d['y'][mask], prediction[mask])})
        return outputs, {'window': width, 'probe': probe, 'n_fit_windows': len(yfit),
                         'iterations': model.n_iter_.tolist(), 'scaler_n_samples_seen': int(scaler.n_samples_seen_),
                         'scaler_mean': scaler.mean_.tolist(), 'converged': True}

    with (nullcontext() if remote else threadpool_limits(limits=1)), ThreadPoolExecutor(max_workers=4) as pool:
        fitted = list(pool.map(fit_one, tasks))
    per_episode = [r for outputs, _ in fitted for r in outputs]
    health = [h for _, h in fitted]
    aggregate = []
    for width, probe in tasks:
        for scope in ('all_valid', 'common_across_windows'):
            rows = [r for r in per_episode if r['role'] == 'eval' and r['window'] == width and r['probe'] == probe and r['frame_scope'] == scope and r['evaluable']]
            aggregate.append({'task': 'PRECONTACT_vs_CONTACT_SUPPORTED', 'window': width, 'probe': probe,
                              'frame_scope': scope, 'BA': float(np.mean([r['BA'] for r in rows])),
                              'macro_F1': float(np.mean([r['macro_F1'] for r in rows])),
                              'per_class_recall': np.mean([r['per_class_recall'] for r in rows], axis=0).tolist(),
                              'n_episodes': len(rows), 'n_windows': sum(r['n_windows'] for r in rows)})
    # One shared resampling index for paired comparisons: independent units are episodes, not frames.
    eval_ids = [r['episode'] for r in coverage if r['role'] == 'eval' and r['evaluable_W15']]
    resamples = np.random.default_rng(20261015).integers(0, len(eval_ids), size=(2000, len(eval_ids)))
    np.save(run / 'bootstrap_episode_indices.npy', resamples)
    contrasts = []
    for width in (1, 9, 15):
        for scope in ('all_valid', 'common_across_windows'):
            for high, low in [('S1', 'S0'), ('S_current', 'S0'), ('S2', 'S1')]:
                if not current_available and high != 'S1':
                    contrasts.append({'window': width, 'frame_scope': scope, 'comparison': f'{high}-{low}', 'status': 'NOT_EVALUABLE_CURRENT_CONSTANT'})
                    continue
                deltas = []
                for ep in eval_ids:
                    pair = {r['probe']: r for r in per_episode if r['role'] == 'eval' and r['episode'] == ep and r['window'] == width and r['frame_scope'] == scope}
                    assert pair[high]['evaluable'] and pair[low]['evaluable']
                    deltas.append(pair[high]['BA'] - pair[low]['BA'])
                contrasts.append({'window': width, 'frame_scope': scope, 'comparison': f'{high}-{low}',
                                  'episode_ids': eval_ids, 'per_episode_delta': deltas, **paired_bootstrap(deltas, resamples)})
    primary = next(r for r in contrasts if r['window'] == 15 and r['frame_scope'] == 'all_valid' and r['comparison'] == 'S1-S0')
    def promising(item):
        return item.get('point_estimate', -1) >= .03 and item.get('positive_episodes', 0) >= math.ceil(2 * len(eval_ids) / 3)
    secondary = next(r for r in contrasts if r['window'] == 15 and r['frame_scope'] == 'all_valid' and r['comparison'] == 'S_current-S0')
    outcome = 'PROMISING_LOAD' if promising(primary) else ('CURRENT_ONLY_LEAD' if promising(secondary) else 'NO_CLEAR_GAIN_IN_PILOT')
    save('metrics.json', {'scope': 'EXPLORATORY_ONLY', 'pilot_outcome': outcome, 'primary': primary,
         'aggregate_eval_episode_equal_weight': aggregate, 'current_evaluable': current_available,
         'frame_weighting_not_used_for_episode_summary': True, 'per_episode': per_episode})
    save('bootstrap.json', {'seed': 20261015, 'replicates': 2000, 'unit': 'episode', 'shared_indices': str(run / 'bootstrap_episode_indices.npy'),
         'formal_significance_claim': False, 'contrasts': contrasts})
    save('confusion_matrices.json', {'class_order': CLASSES, 'per_episode': per_episode})
    save('training_health.json', {'fixed_classifier_fits': len(tasks), 'parallel_workers': 4, 'blas_threads': 1,
                                 'policy_training': False, 'runs': health})
    config.update({'fitted': True, 'fit_health': 'ALL_CONVERGED'})
    save('probe_config.json', config)
    with (OUT / 'per_episode_metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_episode[0]))
        writer.writeheader()
        writer.writerows(per_episode)
    leakage = json.loads((OUT / 'leakage_audit.json').read_text())
    leakage.update({'scaler_fit': True, 'scaler_fit_episode_ids': subset['pilot_fit'], 'eval_never_fit_scaler_or_model': True,
                    'label_freeze': freeze, 'bootstrap_unit': 'episode', 'same_samples_all_probes': True,
                    'common_window_sensitivity_reported': True, 'new_dev_final_read': False})
    save('leakage_audit.json', leakage)
    make_reports(data, subset, fit_rows, per_episode, aggregate, contrasts, primary, outcome, run)
    assert sha(fit_csv) == freeze['fit_csv_sha256'] and sha(eval_csv) == freeze['eval_csv_sha256']
    save('validation.json', {'OVERALL_STATUS': 'PASS', 'PILOT_COMPLETE': True, 'PILOT_OUTCOME': outcome,
         'EVIDENCE_SCOPE': 'EXPLORATORY_ONLY', 'ORIGINAL_F1_COMPLETE': False, 'NEXT_POLICY_TRAINING_AUTHORIZED': False,
         'label_coverage': qualifying, 'all_fixed_probes_converged_finite': True, 'probe_fits': len(tasks),
         'same_samples_per_probe': True, 'fit_only_scaler': True, 'episode_bootstrap_shared_indices': True,
         'labels_unchanged_by_analysis': True, 'official_dev_final_read': False, 'raw': str(run)})
    (run / 'hash_manifest.json').write_text(json.dumps({p.name: sha(p) for p in run.iterdir() if p.is_file()}, indent=2))
    print(json.dumps({'outcome': outcome, 'primary': primary, 'fits': len(tasks), 'raw': str(run)}, ensure_ascii=False))


def make_reports(data, subset, fit_rows, per_episode, aggregate, contrasts, primary, outcome, run):
    figdir = OUT / 'figures'
    (figdir / 'README.md').write_text('Actual pilot-eval window/probe comparisons, paired episode bootstrap CI/deltas, eval position/motion distributions and fit-only precontact-closing vs contact comparison. All exploratory, not causal or real-robot evidence.\n')
    # Fixed visualization of the four existing S0 channels; no new predictive features or thresholds.
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    distribution = []
    for ax, (index, name) in zip(axes.ravel(), [(0, 'pos[t]'), (4, 'delta pos[t]'), (8, 'command[t-1]'), (12, 'delta command[t-1]')]):
        d = [data[ep]['windows'][15] for ep in subset['pilot_eval']]
        x, y = np.concatenate([r['x'][:, index] for r in d]), np.concatenate([r['y'] for r in d])
        bins = np.histogram_bin_edges(x, bins=30)
        for value, label in enumerate(CLASSES):
            z = x[y == value]
            ax.hist(z, bins=bins, alpha=.5, density=True, label=label)
            distribution.append({'channel': name, 'class': label, 'n': len(z), 'min': float(z.min()), 'max': float(z.max()), 'median': float(np.median(z))})
        ax.set_title(name)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figdir / 'position_motion_distribution.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    closing_groups = [[], []]
    for ep in subset['pilot_fit']:
        d = data[ep]['windows'][15]
        closing = np.zeros(len(d['frames']), dtype=bool)
        for r in fit_rows:
            if int(r['episode']) == ep and r['label'] == 'CLOSING':
                closing |= (d['frames'] >= int(r['first_reliable_frame'])) & (d['frames'] <= int(r['last_reliable_frame']))
        closing_groups[0].append(d['x'][(d['y'] == 0) & closing])
        closing_groups[1].append(d['x'][d['y'] == 1])
    closing_groups = [np.concatenate(g) for g in closing_groups]
    for ax, (index, name) in zip(axes.ravel(), [(0, 'pos[t]'), (4, 'delta pos[t]'), (8, 'command[t-1]'), (12, 'delta command[t-1]')]):
        bins = np.histogram_bin_edges(np.concatenate([g[:, index] for g in closing_groups]), bins=30)
        for group, label in zip(closing_groups, ['precontact DURING closing (fit)', 'contact-supported (fit)']):
            ax.hist(group[:, index], bins=bins, density=True, alpha=.5, label=label)
        ax.set_title(name)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figdir / 'precontact_closing_vs_contact_fit.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for probe in COLS:
        for scope, style in [('all_valid', '-'), ('common_across_windows', '--')]:
            rows = sorted([r for r in aggregate if r['probe'] == probe and r['frame_scope'] == scope], key=lambda r: r['window'])
            if rows:
                ax.plot([r['window'] for r in rows], [r['BA'] for r in rows], style, marker='o', label=f'{probe} {scope}')
    ax.set(xlabel='History window (frames)', ylabel='Episode-equal BA', xticks=[1, 9, 15]); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figdir / 'window_probe_comparison.png', dpi=150); plt.close(fig)
    fig, (ax, ci_ax) = plt.subplots(1, 2, figsize=(11, 4))
    main = [r for r in contrasts if r['window'] == 15 and r['frame_scope'] == 'all_valid' and 'point_estimate' in r]
    for i, r in enumerate(main):
        ax.plot(r['episode_ids'], r['per_episode_delta'], marker='o', label=r['comparison'])
        lo, hi = r['percentile_95_ci']
        ci_ax.plot([lo, hi], [i, i], linewidth=3); ci_ax.scatter(r['point_estimate'], i)
    ax.axhline(0, color='gray'); ax.legend(fontsize=8); ax.set(xlabel='Pilot-eval episode', ylabel='BA delta, W15')
    ci_ax.axvline(0, color='gray'); ci_ax.axvline(.03, color='orange', linestyle='--')
    ci_ax.set(yticks=range(len(main)), yticklabels=[r['comparison'] for r in main], xlabel='Paired episode bootstrap 95% CI')
    fig.tight_layout(); fig.savefig(figdir / 'episode_deltas_bootstrap_ci.png', dpi=150); plt.close(fig)
    error_cases = []
    import cv2
    for ep in subset['pilot_eval']:
        if len(error_cases) >= 3:
            break
        with np.load(run / f'pred_ep{ep:03d}_w15_S0.npz') as z:
            bad = np.flatnonzero(z['y'] != z['pred'])
            if not len(bad):
                continue
            j = int(bad[0]); frame = int(z['frames'][j]); truth, pred = int(z['y'][j]), int(z['pred'][j])
        source = OUT / f'evidence_samples/pilot_eval_train_ep{ep:03d}_dual_view.mp4'
        cap = cv2.VideoCapture(str(source)); cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read(); cap.release(); assert ok
        target = OUT / f'evidence_samples/error_S0_train{ep}_frame{frame}.jpg'
        assert cv2.imwrite(str(target), image)
        error_cases.append({'episode': ep, 'frame': frame, 'true': CLASSES[truth], 'predicted': CLASSES[pred], 'image': str(target.relative_to(OUT))})
    (OUT / 'confound_distributions.json').write_text(json.dumps({'descriptive_not_causal': True, 'distributions': distribution, 'S0_error_cases_first_in_episode_order': error_cases}, indent=2))
    s0 = next(r for r in aggregate if r['window'] == 15 and r['probe'] == 'S0' and r['frame_scope'] == 'all_valid')
    confound = ['# 接触与运动混杂审查（探索性）', '',
                f'W15 S0仅凭位置/已知历史命令即可达到episode等权BA={s0["BA"]:.4f}。这是必须一并考虑的基线；不能仅凭S1的高准确率声称接触力感知。', '',
                '原标签中PRECONTACT大多覆盖接近及接触前闭爪，CONTACT_SUPPORTED大多从闭合接触持续至释放，标签与运动阶段、开口及历史命令高度相关的可能性必须保留。位置/运动分布见figures/position_motion_distribution.png及confound_distributions.json；重叠不等于排除混杂，分离也不等于真实接触机制。', '',
                f'专门的接触前闭爪混杂图见figures/precontact_closing_vs_contact_fit.png：复用fit原CLOSING与PRECONTACT交集，共{len(closing_groups[0])}帧，对比fit CONTACT_SUPPORTED共{len(closing_groups[1])}帧的相同位置/运动通道；不用信号推标签、不新增标签。新eval没有阶段标签，不能冒充已独立标了其中的闭爪子集。', '',
                '本合同不追加匹配模型；没有将遮挡段补成真值。错误图按W15 S0每集第一个错帧、episode顺序最多3张固定选择，供检查边界/姿态差异，不按有利案例挑图。', '',
                '人工区间未标不确定段不意味着视觉完全清楚；未标帧被排除，可靠长区间仍有边界/遮挡偏差风险。只有六个独立验证episode，CI只是探索性不确定度，不是正式显著性或等价性证明。']
    (OUT / 'confound_review.md').write_text('\n'.join(confound) + '\n', encoding='utf-8')
    lines = ['# F1P：固定6拟合/6验证接触信号pilot结果', '',
             f'OVERALL_STATUS = PASS  \nPILOT_COMPLETE = true  \nPILOT_OUTCOME = {outcome}  \nEVIDENCE_SCOPE = EXPLORATORY_ONLY  \nORIGINAL_F1_COMPLETE = false  \nNEXT_POLICY_TRAINING_AUTHORIZED = false', '',
             '## 标注与执行', '',
             '复用原Train0/2/4/6/7/9人工标签，只追加原Train10/19/29/38/49/57的两类RGB接触区间；两组六集独立、均通过覆盖门。不读取正式Dev/Final。区间版本在成绩之前冻结并哈希，未知及未标目标排除。12次固定线性分类器拟合在远程隔离CPU环境并行4路、BLAS每路1线程，全部收敛；本地仅重建特征核对输入并汇总预测，无重复拟合、GPU或policy训练。', '',
             '## 固定主结果（W15）', '',
             f'load增量S1−S0：**{primary["point_estimate"]*100:.2f}个百分点**；配对episode bootstrap95%区间[{primary["percentile_95_ci"][0]*100:.2f}, {primary["percentile_95_ci"][1]*100:.2f}]个百分点；{primary["positive_episodes"]}/{primary["n_episodes"]}集为正增量。', '',
             '只有点估计≥3个百分点且至少2/3验证集改善才归PROMISING_LOAD；CI展示不确定度，不用于正式确认。次要current及各窗口全报，不替换主终点。', '',
             '| Task | W | Probe | BA | Macro F1 | Recall(pre/contact) | N episodes | N windows |',
             '|---|---:|---|---:|---:|---|---:|---:|']
    for r in aggregate:
        if r['frame_scope'] == 'all_valid':
            lines.append(f'| contact | {r["window"]} | {r["probe"]} | {r["BA"]:.4f} | {r["macro_F1"]:.4f} | {r["per_class_recall"][0]:.4f}/{r["per_class_recall"][1]:.4f} | {r["n_episodes"]} | {r["n_windows"]} |')
    lines += ['', '### 三项固定W15比较', '']
    for r in main:
        lines.append(f'- {r["comparison"]}: delta={r["point_estimate"]*100:.2f}pp，95%CI=[{r["percentile_95_ci"][0]*100:.2f},{r["percentile_95_ci"][1]*100:.2f}]pp；正向{r["positive_episodes"]}/{r["n_episodes"]}。')
    lines += ['', '## 解释与下一步边界', '',
              '本结果只检验固定线性probe下可解码增量；接触标签可能被位置/闭爪历史解释，详见confound_review.md、分布和错误案例。没有证明P1已使用load/current，也没有证明抓稳、防滑、保持或真实成功率提升。即使有探索线索也不能直接训练Force模块。', '',
              '逐集指标、混淆矩阵、三窗口共同目标敏感性、排除计数与CI见JSON/CSV。六集独立验证的小样本不能排除小收益；NO_CLEAR_GAIN_IN_PILOT不等于NO_INCREMENTAL_SIGNAL。原F1仍未完成，D2/D2.1/G1未闭环与B3实际OC-ACT更正保留。', '',
              '本轮已停止，不增加样本、调参、换窗口或进入F2。下一步仅将此result交用户与ChatGPT验收，讨论是否值得扩大独立验证。']
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (OUT / 'proposal.md').write_text(f'# 下一步仅验收讨论\n\nPilot outcome={outcome}，load主增量{primary["point_estimate"]*100:.2f}pp。先联合主/次要增量、逐集差异、CI宽度和位置/运动混杂审核，决定是否扩大独立标注验证；不建议跳过证据门直接加Force head或训练policy。任何扩大预算或结构实验均待新授权，当前不执行。\n', encoding='utf-8')


if __name__ == '__main__':
    main()
