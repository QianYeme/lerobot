"""Validate saved human first-six labels; execute only the frozen F1 coverage gate."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from run_f1_review_ui import DEFAULT_RESULT, ReviewStore

ROOT = Path(__file__).resolve().parents[1]


def main():
    store = ReviewStore(DEFAULT_RESULT, ROOT / 'outputs/f1_rgb_review_ui/video_cache')
    original = store.csv_path.read_bytes()
    rows = store.read()
    counts = []
    for ep in store.records:
        selected = [r for r in rows if int(r['episode']) == ep and r['label']]
        store.validate(ep, selected)
        frames = {label: set() for label in ('PRECONTACT', 'CONTACT_SUPPORTED', 'CLOSING', 'POST_CLOSE', 'RELEASE')}
        unknown = set()
        for row in selected:
            if row['label'] in ('UNKNOWN', 'OCCLUDED'):
                unknown.update(range(int(row['uncertain_start']), int(row['uncertain_end']) + 1))
            elif row['uncertain_start']:
                unknown.update(range(int(row['uncertain_start']), int(row['uncertain_end']) + 1))
        for row in selected:
            if row['label'] in frames:
                frames[row['label']].update(range(int(row['first_reliable_frame']), int(row['last_reliable_frame']) + 1))
        # Main W15 requires complete causal pos/command-difference history: first target t=16.
        counts_by_label = {label: len({f for f in ids if f >= 16} - unknown) for label, ids in frames.items()}
        counts.append({'episode': ep, 'reviewers': sorted({r['reviewer'] for r in selected}),
                       'rows': len(selected), 'unique_target_frames_W15': counts_by_label,
                       'unknown_frames_excluded': len(unknown),
                       'contact_both_classes_at_least_5': all(counts_by_label[k] >= 5 for k in ('PRECONTACT', 'CONTACT_SUPPORTED'))})
    qualifying = sum(r['contact_both_classes_at_least_5'] for r in counts)
    gate = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'csv_sha256': hashlib.sha256(original).hexdigest(), 'source': 'HUMAN_REVIEWED',
            'basis': 'User confirmed annotation complete; saved reviewer and RGB evidence descriptions',
            'codex_independent_continuous_video_review': False,
            'first_six': counts, 'qualifying_contact_episodes': qualifying, 'threshold': 4,
            'decision': 'CONTINUE_TRAIN36_CONTACT_REVIEW' if qualifying >= 4 else 'SWITCH_TO_TRAIN36_PHASE_REVIEW',
            'task_frozen': False, 'train_total': 36, 'train_reviewed': 6,
            'train_remaining': 30, 'dev_label_review_authorized_now': False,
            'probe_fit_authorized_now': False, 'policy_training_authorized': False,
            'limitation': 'Contact-supported does not establish calibrated force, stable hold or exact lift-off; no F1 scores computed'}
    run = ROOT / 'results/experiments/F1_FORMAL4_CONTACT_SIGNAL_FEASIBILITY' / (
        'label_gate_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    run.mkdir(parents=True)
    (run / 'review_intervals_human_snapshot.csv').write_bytes(original)
    shutil.copy2(__file__, run / Path(__file__).name)
    (run / 'gate.json').write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding='utf-8')
    for name in ('first_six_label_gate.json', 'label_coverage.json', 'task_freeze.json'):
        path = DEFAULT_RESULT / name
        if path.exists():
            shutil.copy2(path, run / ('previous_' + name))
    gate['raw_evidence'] = str(run)
    (DEFAULT_RESULT / 'first_six_label_gate.json').write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding='utf-8')
    coverage = {'status': 'FIRST_SIX_GATE_EVALUATED_TRAIN36_PENDING', 'train_reviewed': 6, 'train_remaining': 30,
                'dev_reviewed': 0, 'first_six_counts': counts, 'contact_gate': qualifying >= 4,
                'full_train_task_threshold': 'at least 24/36 episodes, both classes >=5 reliable targets at W15',
                'raw_evidence': str(run)}
    (DEFAULT_RESULT / 'label_coverage.json').write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding='utf-8')
    freeze = {'status': 'NOT_FROZEN', 'first_six_gate': gate['decision'],
              'reason': 'First-six human contact gate evaluated; remaining Train30 review required before Train task/label freeze',
              'train_first_six_manifest_order': list(store.records), 'dev_labels_reviewed': False,
              'label_csv_snapshot_sha256': gate['csv_sha256']}
    (DEFAULT_RESULT / 'task_freeze.json').write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding='utf-8')
    assert store.csv_path.read_bytes() == original
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
