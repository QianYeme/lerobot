"""F1Q matching and balance ONLY; freeze configuration before opening matched outcomes."""
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'experiments/force/f1p_contact_signal_pilot/index1/result'
BASE = ROOT / 'experiments/force/f1q_load_confound_control/index1'
NAMES = ['gripper.pos', 'delta_gripper.pos', 'command_t_minus_1', 'delta_command_t_minus_1']
MATCH_COLS = [0, 4, 8, 12]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def greedy_match(x, y, frames, maximum=.75):
    """Contact rows in source order; nearest unused precontact; earlier frame breaks ties."""
    pre = np.flatnonzero(y == 0)
    pre = pre[np.argsort(frames[pre], kind='stable')]
    contact = np.flatnonzero(y == 1)
    contact = contact[np.argsort(frames[contact], kind='stable')]
    available = np.ones(len(pre), dtype=bool)
    pairs = []
    for c in contact:
        candidates = np.flatnonzero(available)
        if not len(candidates):
            break
        distance = np.linalg.norm(x[pre[candidates]] - x[c], axis=1)
        best = int(np.argmin(distance))
        j = candidates[best]
        if distance[best] <= maximum:
            pairs.append((int(pre[j]), int(c), float(distance[best])))
            available[j] = False
    return pairs


def balance(x, y):
    result = []
    for j, name in enumerate(NAMES):
        a, b = x[y == 0, j], x[y == 1, j]
        if len(a) < 2 or len(b) < 2:
            result.append({'variable': name, 'pre_mean': None, 'contact_mean': None,
                           'pre_std': None, 'contact_std': None, 'SMD': None, 'balance_pass': False})
            continue
        s0, s1 = float(a.std(ddof=1)), float(b.std(ddof=1))
        denominator = np.sqrt((s0*s0 + s1*s1) / 2)
        difference = float(b.mean() - a.mean())
        smd = difference / denominator if denominator > 0 else (0. if difference == 0 else None)
        result.append({'variable': name, 'pre_mean': float(a.mean()), 'contact_mean': float(b.mean()),
                       'pre_std': s0, 'contact_std': s1, 'SMD': smd,
                       'balance_pass': smd is not None and abs(smd) <= .20})
    return result


def main():
    out = BASE / 'result'
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Existing F1Q outputs: do not overwrite or rematch after results')
    out.mkdir(parents=True)
    raw = ROOT / 'results/experiments/F1Q_LOAD_CONFOUND_CONTROL' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    raw.mkdir(parents=True)

    def save(name, obj):
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    config = {'window': 15, 'matching_columns_in_F1P_features': MATCH_COLS, 'matching_variables': NAMES,
              'scaling_population': 'all reliable W15 Pilot-fit targets only, before matching',
              'metric': 'Euclidean', 'maximum_distance': .75, 'replacement': False, 'within_episode': True,
              'contact_iteration_order': 'ascending original source frame; deterministic traversal, NOT a distance feature',
              'ties': 'smaller PRECONTACT source frame', 'minimum_pairs': 20, 'minimum_episodes_each_group': 4,
              'balance_SMD_max_abs': .20, 'SMD_definition': '(contact_mean-pre_mean)/sqrt((sample_var_pre+sample_var_contact)/2)',
              'SMD_zero_variance': 'zero if means equal, otherwise undefined and fails balance',
              'primary_balance_population': 'pooled matched eval episodes meeting >=20 pairs',
              'additional_audit_population': 'all matched episodes including coverage failures; report separately',
              'probe_features': {'M0_matched': list(range(16)), 'M1_matched': list(range(20))},
              'probe_scaler_population': 'matched Pilot-fit only', 'probe_seed': 20261015,
              'frozen_before_matching_and_probe_performance': True,
              'decision_scope': 'EXPLORATORY_ONLY', 'raw': str(raw)}
    save('matching_config.json', config)
    shutil.copy2(out / 'matching_config.json', raw / 'matching_config_frozen.json')
    subset = json.loads((PARENT / 'subset_manifest.json').read_text())
    freeze = json.loads((PARENT / 'label_freeze.json').read_text())
    source = Path(freeze['raw'])
    assert sha(PARENT / 'subset_manifest.json') == freeze['subset_sha256']
    assert sha(PARENT / 'evidence_samples/pilot_fit_labels_snapshot.csv') == freeze['fit_csv_sha256']
    assert sha(PARENT / 'evidence_samples/review_intervals.csv') == freeze['eval_csv_sha256']
    assert not set(subset['pilot_fit']) & set(subset['pilot_eval'])
    save('subset_manifest.json', {'pilot_fit': subset['pilot_fit'], 'pilot_eval': subset['pilot_eval'],
         'parent_subset_sha256': sha(PARENT / 'subset_manifest.json'), 'label_freeze': freeze,
         'official_dev_final_read': False, 'new_labels': False})
    save('provenance.json', {'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         'status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
         'parent_features': str(source), 'parent_scientific_metrics_read_by_this_script': False,
         'contract_sha256': sha(BASE / 'decision_F1Q_LOAD_CONFOUND_CONTROL.md'),
         'script_sha256': sha(Path(__file__)), 'cli': 'python scripts/match_f1q_confound.py', 'raw': str(raw)})
    shutil.copy2(__file__, raw / Path(__file__).name)
    shutil.copy2(BASE / 'decision_F1Q_LOAD_CONFOUND_CONTROL.md', raw / 'decision_frozen.md')
    (raw / 'start_dirty.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary'], cwd=ROOT))
    records = {}
    for ep in subset['pilot_fit'] + subset['pilot_eval']:
        path = source / f'features_ep{ep:03d}_w15.npz'
        with np.load(path) as z:
            records[ep] = {k: z[k].copy() for k in ['frames', 'x', 'y']}
        assert records[ep]['x'].shape[1] == 24 and np.isfinite(records[ep]['x']).all()
    scaler = StandardScaler().fit(np.concatenate([records[ep]['x'][:, MATCH_COLS] for ep in subset['pilot_fit']]))
    save('matching_scaler.json', {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
         'n_samples_seen': int(scaler.n_samples_seen_), 'fit_episodes': subset['pilot_fit']})
    coverage, audit, pair_rows = [], [], []
    for role, ids in [('fit', subset['pilot_fit']), ('eval', subset['pilot_eval'])]:
        for ep in ids:
            d = records[ep]
            match_x = d['x'][:, MATCH_COLS]
            pairs = greedy_match(scaler.transform(match_x), d['y'], d['frames'])
            indices = np.array([ix for a, b, _ in pairs for ix in (a, b)], dtype=int)
            d['matched_indices'] = indices
            np.savez_compressed(raw / f'matched_ep{ep:03d}.npz', x=d['x'][indices, :20], y=d['y'][indices], frames=d['frames'][indices])
            distances = [v for _, _, v in pairs]
            coverage.append({'role': role, 'episode': ep, 'pre_before': int((d['y'] == 0).sum()),
                             'contact_before': int((d['y'] == 1).sum()), 'matched_pairs': len(pairs),
                             'valid': len(pairs) >= 20, 'distance_mean': float(np.mean(distances)) if distances else None,
                             'distance_max': max(distances) if distances else None,
                             'distance_q50_q90': np.quantile(distances, [.5, .9]).tolist() if distances else None,
                             'exclusion_reason': '' if len(pairs) >= 20 else 'fewer than 20 pairs at frozen distance<=0.75'})
            for a, b, v in pairs:
                pair_rows.append({'role': role, 'episode': ep, 'pre_frame': int(d['frames'][a]),
                                  'contact_frame': int(d['frames'][b]), 'distance': v})
            for stage, xx, yy in [('before', match_x, d['y']), ('after', match_x[indices], d['y'][indices])]:
                audit.extend([{'role': role, 'episode': str(ep), 'stage': stage, **r} for r in balance(xx, yy)])
    qualifying = {role: [r['episode'] for r in coverage if r['role'] == role and r['valid']] for role in ['fit', 'eval']}
    for role in ['fit', 'eval']:
        for group, ids in [('all', subset['pilot_fit' if role == 'fit' else 'pilot_eval']), ('valid', qualifying[role])]:
            for stage in ['before', 'after']:
                xs, ys = [], []
                for ep in ids:
                    d = records[ep]
                    ix = np.arange(len(d['y'])) if stage == 'before' else d['matched_indices']
                    xs.append(d['x'][ix][:, MATCH_COLS]); ys.append(d['y'][ix])
                if xs:
                    audit.extend([{'role': role, 'episode': f'POOLED_{group}', 'stage': stage, **r}
                                  for r in balance(np.concatenate(xs), np.concatenate(ys))])
    coverage_pass = min(map(len, qualifying.values())) >= 4
    primary_balance = [r for r in audit if r['role'] == 'eval' and r['episode'] == 'POOLED_valid' and r['stage'] == 'after']
    balance_pass = len(primary_balance) == 4 and all(r['balance_pass'] for r in primary_balance)
    outcome = 'UNRESOLVED_MATCHING_COVERAGE' if not coverage_pass else ('UNRESOLVED_RESIDUAL_CONFOUND' if not balance_pass else 'READY_FOR_FIXED_PROBES')
    gate = {'coverage_pass': coverage_pass, 'qualifying': qualifying, 'pooled_eval_balance_pass': balance_pass,
            'outcome_before_any_probe_performance': outcome, 'new_probe_performance_read': False,
            'probe_fits': 0, 'raw': str(raw)}
    save('matching_gate.json', gate)
    for name, rows in [('matching_coverage.csv', coverage), ('balance_audit.csv', audit)]:
        with (out / name).open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (raw / 'matched_pairs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['role', 'episode', 'pre_frame', 'contact_frame', 'distance']); writer.writeheader(); writer.writerows(pair_rows)
    figdir = out / 'figures'; figdir.mkdir()
    for j, name in enumerate(NAMES):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        for ax, stage in zip(axes, ['before', 'after']):
            xs, ys = [], []
            for ep in subset['pilot_eval']:
                d = records[ep]; ix = np.arange(len(d['y'])) if stage == 'before' else d['matched_indices']
                xs.extend(d['x'][ix, MATCH_COLS[j]]); ys.extend(d['y'][ix])
            xs, ys = np.asarray(xs), np.asarray(ys)
            if len(xs):
                bins = np.histogram_bin_edges(xs, bins=25)
                for label, title in [(0, 'PRECONTACT'), (1, 'CONTACT_SUPPORTED')]:
                    ax.hist(xs[ys == label], bins=bins, alpha=.5, density=True, label=title)
            ax.set_title(name + ' ' + stage); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(figdir / f'matching_variable_{j}.png', dpi=150); plt.close(fig)
    fig, (ax, dist) = plt.subplots(1, 2, figsize=(10, 3))
    ax.bar([str(r['episode']) for r in coverage], [r['matched_pairs'] for r in coverage]); ax.axhline(20, color='red', linestyle='--')
    ax.set(xlabel='fit six then eval six', ylabel='Matched pairs')
    if pair_rows:
        dist.hist([r['distance'] for r in pair_rows], bins=25)
    dist.axvline(.75, color='red', linestyle='--'); dist.set(xlabel='Standardized matching distance', ylabel='Pairs')
    fig.tight_layout(); fig.savefig(figdir / 'coverage_distance.png', dpi=150); plt.close(fig)
    print(json.dumps(gate))


if __name__ == '__main__':
    main()
