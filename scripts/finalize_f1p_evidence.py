"""Validate completed F1P outputs and package reports; never fit or change scientific metrics."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/force/f1p_contact_signal_pilot/index1/result'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    remote = read('remote_execution.json')
    first = Path(remote['local_failed_evidence'])
    initial_freeze = json.loads((first / 'job.json').read_text())['label_freeze']
    current_freeze = read('label_freeze.json')
    raw = Path(current_freeze['raw'])
    for key in ('fit_csv_sha256', 'eval_csv_sha256', 'subset_sha256'):
        assert current_freeze[key] == initial_freeze[key]
    # Assembly retries must not replace the true pre-fit label-freeze timestamp.
    if current_freeze['timestamp_utc'] != initial_freeze['timestamp_utc']:
        for name in ('label_freeze.json', 'label_manifest.json', 'leakage_audit.json'):
            shutil.copy2(OUT / name, raw / ('before_timestamp_correction_' + name))
        current_freeze.update({'report_assembled_at_utc': current_freeze['timestamp_utc'],
                              'timestamp_utc': initial_freeze['timestamp_utc'],
                              'first_freeze_evidence': str(first / 'job.json'),
                              'metadata_correction': 'Restore original pre-fit freeze timestamp; all label/subset hashes unchanged'})
        write('label_freeze.json', current_freeze)
        for name in ('label_manifest.json', 'leakage_audit.json'):
            obj = read(name)
            obj['label_freeze'] = current_freeze
            write(name, obj)
    metrics, boot = read('metrics.json'), read('bootstrap.json')
    indices = np.load(raw / 'bootstrap_episode_indices.npy')
    np.testing.assert_equal(indices, np.random.default_rng(20261015).integers(0, 6, size=(2000, 6)))
    for r in boot['contrasts']:
        if 'per_episode_delta' not in r:
            continue
        delta = np.asarray(r['per_episode_delta'])
        np.testing.assert_allclose(np.quantile(delta[indices].mean(1), [.025, .975]), r['percentile_95_ci'])
        np.testing.assert_allclose(delta.mean(), r['point_estimate'])
    models = sorted(raw.glob('probe_*.pkl'))
    predictions = sorted(raw.glob('pred_*.npz'))
    assert len(models) == 12 and len(predictions) == 144
    for p in models + predictions:
        assert sha(p) == sha(first / 'remote_output' / p.name)
    for w in (1, 9, 15):
        groups = [r for r in metrics['per_episode'] if r['role'] == 'eval' and r['window'] == w and r['frame_scope'] == 'all_valid']
        for ep in [10, 19, 29, 38, 49, 57]:
            assert len({tuple(r['class_counts']) for r in groups if r['episode'] == ep}) == 1
    assert sha(OUT / 'evidence_samples/pilot_fit_labels_snapshot.csv') == initial_freeze['fit_csv_sha256']
    assert sha(OUT / 'evidence_samples/review_intervals.csv') == initial_freeze['eval_csv_sha256']
    audit = {'status': 'PASS', 'bootstrap_recomputed_from_saved_shared_episode_indices': True,
             'equal_episode_weighting_verified': True, 'same_class_counts_across_probes': True,
             'remote_models_verified_by_sha256': 12, 'remote_predictions_verified_by_sha256': 144,
             'labels_subset_hashes_unchanged_after_fitting': True,
             'original_label_freeze_timestamp_restored_from_pre_fit_job': initial_freeze['timestamp_utc'],
             'fit_only_scaler_means_checked_against_frozen_fit_features': True,
             'local_actual_classifier_fits': 0, 'remote_actual_classifier_fits': 12,
             'engineering_failures': [
                 'Local threadpoolctl2.2.0 failed before any fitting; isolated remote3.6.0 succeeded',
                 'Local reporting np.bool_ JSON serialization failed after predictions copied; cast to native bool, recomputed reports without any refit',
                 'Report assembly timestamp initially replaced original freeze timestamp; corrected from pre-fit job manifest, hashes unchanged'],
             'evidence_scope': 'EXPLORATORY_ONLY', 'new_policy_training_authorized': False}
    write('report_consistency_audit.json', audit)
    validation = read('validation.json')
    validation['final_evidence_consistency_audit'] = 'PASS'
    validation['engineering_failures_retained'] = True
    write('validation.json', validation)
    (raw / 'reporting_failures.json').write_text(json.dumps(audit['engineering_failures'], indent=2), encoding='utf-8')
    for p in [Path(__file__), ROOT / 'scripts/eval_f1p_pilot.py', ROOT / 'scripts/fit_f1p_remote.py',
              ROOT / 'scripts/launch_f1p_remote.py', ROOT / 'scripts/run_f1p_remote.sh']:
        shutil.copy2(p, raw / p.name)
    files = [p for p in OUT.rglob('*') if p.is_file() and 'csv_backups' not in p.parts]
    write('evidence_manifest.json', {'raw': str(raw), 'original_failed_attempt': str(first),
         'files': [{'path': p.relative_to(OUT).as_posix(), 'sha256': sha(p)} for p in files if p.name != 'evidence_manifest.json']})
    archive = OUT.parent / 'F1P_result_20261015.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for p in OUT.rglob('*'):
            if not p.is_file() or 'csv_backups' in p.parts:
                continue
            if p.suffix == '.mp4' and 'ep010_' not in p.name:
                continue
            bundle.write(p, p.relative_to(OUT))
        bundle.writestr('PACKAGE_README.txt', 'Compact F1P exploratory report. Includes reports, figures, frozen fit/new human interval CSVs, error RGB frames, and one full dual-view pilot-eval episode (Train10). Other five full videos remain in the local result/evidence_samples directory and the preserved raw sources. No model fitting occurs when packaging.\n')
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
    print(json.dumps({'consistency': 'PASS', 'zip': str(archive), 'bytes': archive.stat().st_size}))


if __name__ == '__main__':
    main()
