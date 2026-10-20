"""Frozen F1RM retrospective matching. No training, robot access or threshold search."""
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/force/f1rm_teleop_load_retrospective_match'
OUT = BASE / 'index1/result'
NAMES = ['pos', 'delta_pos', 'previous_command', 'delta_previous_command']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def table(path, rows, columns=None):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def features(state, action):
    return np.column_stack([state[2:, 5], np.diff(state[:, 5])[1:],
                            action[1:-1, 5], np.diff(action[:, 5])[:-1]])


def validate_annotations(rows, ids):
    occupied = {name: set() for name in ids}
    for row in rows:
        trial = row['trial_id']
        if trial not in occupied:
            raise ValueError('Unknown trial: ' + trial)
        allowed = {'FREE_CLOSE_CLOSING', 'UNKNOWN'} if 'kind1' in trial else {'PRECONTACT', 'CONTACT', 'UNKNOWN'}
        if row['label'] not in allowed or not row['reviewer'].strip() or not row['reason'].strip():
            raise ValueError('Invalid label/reviewer/reason')
        a, b = int(row['first_reliable_frame']), int(row['last_reliable_frame'])
        interval = set(range(a, b + 1))
        if not 0 <= a <= b <= 149 or occupied[trial] & interval:
            raise ValueError('Invalid or overlapping interval: ' + trial)
        occupied[trial].update(interval)
    if any(not frames for frames in occupied.values()):
        raise ValueError('All ten trials must be reviewed')


def greedy(source, target, maximum):
    available = np.ones(len(source), dtype=bool)
    pairs = []
    for j, x in enumerate(target):
        candidates = np.flatnonzero(available)
        if not len(candidates):
            break
        distances = np.linalg.norm(source[candidates] - x, axis=1)
        best = int(np.argmin(distances))
        if distances[best] <= maximum:
            i = int(candidates[best])
            available[i] = False
            pairs.append((i, j, float(distances[best])))
    return pairs


def smd(source, target):
    if len(source) < 2 or len(target) < 2:
        return None
    delta = float(np.mean(target) - np.mean(source))
    denominator = float(np.sqrt((np.var(source, ddof=1) + np.var(target, ddof=1)) / 2))
    return delta / denominator if denominator else (0.0 if delta == 0 else None)


def describe(values):
    if not len(values):
        return dict(n=0, mean=None, median=None, std=None, iqr=None, positive_fraction=None)
    values = np.asarray(values)
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
                std=float(values.std(ddof=1)) if len(values) > 1 else None,
                iqr=float(np.subtract(*np.percentile(values, [75, 25]))),
                positive_fraction=float(np.mean(values > 0)))


def cluster_bootstrap(groups, resamples=2000, seed=20261016):
    if len(groups) < 2:
        return dict(status='NOT_ESTIMABLE_FEWER_THAN_TWO_CLUSTERS', ci95=None)
    rng = np.random.default_rng(seed)
    values = [np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))]).mean()
              for _ in range(resamples)]
    return dict(status='EXPLORATORY_CONDITIONAL_ON_FIXED_MATCHES', clusters=len(groups),
                resamples=resamples, seed=seed, ci95=np.percentile(values, [2.5, 97.5]).tolist(),
                method='resample CUP trials with replacement; keep all pairs in each sampled cluster',
                limitation='conditions on observed FREE pool; not two-way-cluster or rematching uncertainty')


def main():
    if (OUT / 'label_freeze.json').exists():
        raise RuntimeError('Already frozen/executed: do not overwrite or rematch')
    source_csv = BASE / 'contact_annotations.csv'
    manifest = list(csv.DictReader((OUT / 'trial_manifest.csv').open(encoding='utf-8-sig')))
    rows = list(csv.DictReader(source_csv.open(encoding='utf-8-sig')))
    ids = sorted(row['trial_id'] for row in manifest)
    validate_annotations(rows, ids)
    config = json.loads((OUT / 'matching_config.json').read_text(encoding='utf-8'))
    assert config['primary_distance'] == .75 and config['sensitivity_distance'] == 1
    raw = ROOT / 'results/experiments/F1RM_TELEOP_LOAD_RETROSPECTIVE_MATCH' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    raw.mkdir(parents=True)
    for name in ['summary.md', 'validation.md', 'metrics.json', 'contact_annotations.csv', 'leakage_audit.md', 'proposal.md']:
        shutil.copy2(OUT / name, raw / ('before_' + name))
    shutil.copy2(source_csv, OUT / 'contact_annotations.csv')
    shutil.copy2(source_csv, raw / 'independent_annotations.csv')
    shutil.copy2(OUT / 'matching_config.json', raw / 'matching_config.json')
    shutil.copy2(__file__, raw / 'analyze_f1rm.py')
    # Freeze exact user labels before any force effects or matching outcomes.
    save(OUT / 'label_freeze.json', dict(source=str(source_csv), sha256=sha(source_csv), rows=len(rows),
         reviewer='kg', provenance='USER_SUBMITTED_RGB_REVIEW', raw=str(raw),
         warning='All reasons are literal 1; substantive per-interval visual rationale not supplied',
         unknown_ep001_preserved=True, config_sha256=sha(OUT / 'matching_config.json')))
    validation = json.loads((OUT / 'input_validation.json').read_text(encoding='utf-8'))
    for name, expected in validation['original_file_sha256'].items():
        assert sha(ROOT / name) == expected, name
    records, arrays = {}, {}
    for kind in ['formal5_kind1', 'formal5_kind2']:
        dataset = ROOT / '数据集/formal5' / kind
        info = json.loads((dataset / 'meta/info.json').read_text(encoding='utf-8'))
        assert info['features']['observation.state']['names'][5:8] == ['gripper.pos', 'gripper.load', 'gripper.curr']
        assert info['features']['action']['names'][5] == 'gripper.pos'
        df = pd.read_parquet(dataset / 'data/chunk-000/file-000.parquet')
        for ep in range(5):
            name = f'{kind}_ep{ep:03d}'
            d = df[df.episode_index == ep].sort_values('frame_index')
            state, action = np.stack(d['observation.state']), np.stack(d.action)
            assert state.shape == (150, 9) and action.shape == (150, 6)
            assert np.isfinite(state).all() and np.isfinite(action).all()
            assert np.array_equal(d.frame_index, np.arange(150))
            assert np.allclose(np.diff(d.timestamp), 1 / 30, atol=1e-6)
            arrays[name] = state
            labels = np.full(150, 'UNANNOTATED', dtype=object)
            for row in rows:
                if row['trial_id'] == name:
                    labels[int(row['first_reliable_frame']):int(row['last_reliable_frame'])+1] = row['label']
            x = features(state, action)
            for frame in range(2, 150):
                records[(name, frame)] = dict(trial_id=name, frame=frame, label=labels[frame], x=x[frame-2])
    free = [r for r in records.values() if r['label'] == 'FREE_CLOSE_CLOSING']
    contact = [r for r in records.values() if r['label'] == 'CONTACT']
    pre = [r for r in records.values() if r['label'] == 'PRECONTACT']
    fit = np.array([r['x'] for r in free + pre])
    mean, scale = fit.mean(axis=0), fit.std(axis=0)
    scale[scale == 0] = 1
    save(OUT / 'matching_scaler.json', dict(features=NAMES, mean=mean.tolist(), scale=scale.tolist(),
         n_fit=len(fit), free_closing_frames=len(free), cup_precontact_frames=len(pre), contact_frames_in_fit=0,
         fit_frame_ids=[f"{r['trial_id']}:{r['frame']}" for r in free + pre]))
    sx, tx = np.array([r['x'] for r in free]), np.array([r['x'] for r in contact])
    passes, coverage, balances, per_trial, pair_records = {}, [], [], [], {}
    pair_columns = ['pass', 'free_trial', 'free_frame', 'cup_trial', 'cup_frame', 'distance',
                    'free_load', 'cup_load', 'delta_load', 'free_current', 'cup_current', 'delta_current']
    cup_ids = [name for name in ids if 'kind2' in name]
    for name, threshold in [('primary', .75), ('sensitivity', 1.0)]:
        if name == 'sensitivity' and passes['primary']['coverage_pass']:
            table(OUT / 'matching_pairs_sensitivity.csv', [], pair_columns)
            passes[name] = dict(status='NOT_RUN_PRIMARY_COVERAGE_PASSED')
            continue
        pairs = greedy((sx-mean)/scale, (tx-mean)/scale, threshold)
        # Only now access load/current for the fixed, non-force-selected pairs.
        pair_rows = []
        for i, j, distance in pairs:
            f, c = free[i], contact[j]
            fs, cs = arrays[f['trial_id']][f['frame']], arrays[c['trial_id']][c['frame']]
            pair_rows.append(dict(zip(pair_columns, [name, f['trial_id'], f['frame'], c['trial_id'], c['frame'],
                distance, float(fs[6]), float(cs[6]), float(cs[6]-fs[6]),
                float(fs[7]), float(cs[7]), float(cs[7]-fs[7])])))
        table(OUT / f'matching_pairs_{name}.csv', pair_rows, pair_columns)
        pair_records[name] = pair_rows
        counts = []
        for trial in cup_ids:
            eligible = sum(r['trial_id'] == trial for r in contact)
            group = [r for r in pair_rows if r['cup_trial'] == trial]
            n = len(group)
            counts.append(n)
            distances = [r['distance'] for r in group]
            coverage.append(dict(pass_name=name, trial_id=trial, eligible_contact=eligible, matched_pairs=n,
                matched_free_frames=n, coverage=n/eligible if eligible else None, qualifying=n>=10,
                distance_mean=float(np.mean(distances)) if n else None,
                distance_p95=float(np.percentile(distances,95)) if n else None))
            stats = describe([r['delta_load'] for r in group])
            per_trial.append(dict(pass_name=name, trial_id=trial, **stats,
                current_mean=describe([r['delta_current'] for r in group])['mean'],
                distance_mean=float(np.mean(distances)) if n else None,
                distance_p95=float(np.percentile(distances,95)) if n else None))
        ix, jx = [i for i,j,d in pairs], [j for i,j,d in pairs]
        audit = []
        for k, variable in enumerate(NAMES):
            after = smd(sx[ix,k], tx[jx,k])
            row = dict(pass_name=name, variable=variable, before_smd=smd(sx[:,k],tx[:,k]), after_smd=after,
                       balance_pass=after is not None and abs(after)<=.20)
            balances.append(row)
            audit.append(row)
        passes[name] = dict(threshold=threshold, pairs=len(pairs), eligible_contact=len(contact),
            free_pool=len(free), per_trial_counts=counts, qualifying_trials=sum(n>=10 for n in counts),
            coverage_pass=sum(n>=10 for n in counts)>=3 and len(pairs)>=40,
            balance_pass=all(r['balance_pass'] for r in audit),
            load=describe([r['delta_load'] for r in pair_rows]),
            current_secondary=describe([r['delta_current'] for r in pair_rows]))
    table(OUT / 'matching_coverage.csv', coverage)
    table(OUT / 'balance_audit.csv', balances)
    table(OUT / 'per_trial_metrics.csv', per_trial)
    bootstrap = {}
    for name, pairs in pair_records.items():
        bootstrap[name] = {signal: cluster_bootstrap([np.array([r['delta_'+signal] for r in pairs if r['cup_trial']==trial])
            for trial in cup_ids if any(r['cup_trial']==trial for r in pairs)]) for signal in ['load','current']}
    save(OUT / 'bootstrap.json', bootstrap)
    figures = OUT / 'figures'
    figures.mkdir(exist_ok=True)
    def finish(name):
        plt.tight_layout()
        plt.savefig(figures / name, dpi=160)
        plt.close()
    for matched in [False, True]:
        plt.figure(figsize=(8,5))
        for population, color in [(free,'tab:blue'),(contact,'tab:orange')]:
            if matched:
                prefix = 'free' if population is free else 'cup'
                keys = [(r[prefix+'_trial'],r[prefix+'_frame']) for r in pair_records['primary']]
                values = [arrays[t][f] for t,f in keys]
            else:
                values = [arrays[r['trial_id']][r['frame']] for r in population]
            if values:
                a = np.array(values)
                plt.scatter(a[:,5],a[:,6],s=12,alpha=.5,color=color,label='FREE_CLOSE' if population is free else 'CONTACT')
        plt.xlabel('gripper.pos (native units)'); plt.ylabel('gripper.load (raw units)'); plt.legend()
        plt.title('Primary matched (descriptive)' if matched else 'Eligible raw populations (descriptive)')
        finish('load_vs_position_matched.png' if matched else 'load_vs_position_raw.png')
    for filename, field, xlabel in [('matched_load_differences.png','delta_load','Cup minus free load (raw units)'),
                                     ('matching_distance_distribution.png','distance','Standardized Euclidean distance')]:
        plt.figure(figsize=(8,4))
        for name,pairs in pair_records.items():
            plt.hist([r[field] for r in pairs],bins=18,alpha=.5,label=name)
        plt.xlabel(xlabel); plt.ylabel('Pairs (not independent trials)'); plt.legend(); finish(filename)
    primary_rows = pair_records['primary']
    plt.figure(figsize=(6,5)); plt.scatter([r['free_load'] for r in primary_rows],[r['cup_load'] for r in primary_rows],s=12)
    plt.xlabel('Free load'); plt.ylabel('Cup load'); finish('matched_pair_scatter.png')
    plt.figure(figsize=(8,4))
    for name in pair_records:
        r = [r for r in per_trial if r['pass_name']==name and r['n']]
        plt.plot([x['trial_id'][-3:] for x in r],[x['mean'] for x in r],'o-',label=name)
    plt.axhline(0,color='gray');plt.ylabel('Mean cup minus free load');plt.xlabel('Cup trial');plt.legend();finish('per_trial_residual_load.png')
    plt.figure(figsize=(9,4))
    for name in pair_records:
        b = [r for r in balances if r['pass_name']==name]
        plt.plot(NAMES,[r['after_smd'] for r in b],'o-',label=name+' after')
    plt.plot(NAMES,[r['before_smd'] for r in balances[:4]],'o--',label='before')
    plt.axhline(.2,color='gray');plt.axhline(-.2,color='gray');plt.ylabel('SMD');plt.legend();finish('balance_before_after.png')
    onset_rows = []
    plt.figure(figsize=(9,5))
    for trial in cup_ids:
        contact_rows = [r for r in rows if r['trial_id']==trial and r['label']=='CONTACT']
        if not contact_rows:
            onset_rows.append(dict(trial_id=trial,status='NO_RELIABLE_CONTACT',onset=None,post_minus_pre=None))
            continue
        onset = min(int(r['first_reliable_frame']) for r in contact_rows)
        before = [f for f in range(max(0,onset-15),onset) if records.get((trial,f),{}).get('label')=='PRECONTACT']
        after = [f for f in range(onset,min(150,onset+15)) if records.get((trial,f),{}).get('label')=='CONTACT']
        value = float(arrays[trial][after,6].mean()-arrays[trial][before,6].mean()) if before and after else None
        onset_rows.append(dict(trial_id=trial,status='ANNOTATED_BOUNDARY_DIAGNOSTIC' if value is not None else 'INSUFFICIENT_RELIABLE_WINDOW',
                               onset=onset,post_minus_pre=value))
        frame = np.arange(max(0,onset-15),min(150,onset+15))
        plt.plot((frame-onset)/30,arrays[trial][frame,6],label=trial[-3:])
    plt.axvline(0,color='gray');plt.xlabel('Seconds from first annotated CONTACT');plt.ylabel('Raw load');plt.legend();finish('contact_aligned_load.png')
    table(OUT / 'contact_onset_diagnostic.csv',onset_rows)
    primary = passes['primary']
    if not primary['coverage_pass']:
        outcome = 'UNRESOLVED_MATCHING_COVERAGE'
    elif not primary['balance_pass']:
        outcome = 'UNRESOLVED_RESIDUAL_CONFOUND'
    else:
        # Conservative: no promotion on frame pooling alone or an undefined dominance rule.
        outcome = 'WEAK_TELEOP_RESIDUAL_SIGNAL' if primary['load']['mean']>0 else 'NO_CLEAR_TELEOP_RESIDUAL_GAIN'
    leave_one_out = {trial:describe([r['delta_load'] for r in primary_rows if r['cup_trial']!=trial])['mean'] for trial in cup_ids}
    contributions = {trial:float(sum(r['delta_load'] for r in primary_rows if r['cup_trial']==trial)) for trial in cup_ids}
    save(OUT / 'metrics.json',dict(OVERALL_STATUS='PASS',F1RM_COMPLETE=True,SCIENTIFIC_OUTCOME=outcome,
         EVIDENCE_SCOPE='EXPLORATORY_ONLY',passes=passes,primary_leave_one_cup_trial_out_means=leave_one_out,
         primary_signed_residual_sums_by_trial=contributions,contact_onset=onset_rows,
         label_sha256=sha(source_csv),raw=str(raw),scientific_positive_claim_authorized=False))
    for path in OUT.iterdir():
        if path.is_file() and path.suffix in ['.json','.csv']:
            shutil.copy2(path,raw/path.name)
    print(json.dumps(dict(outcome=outcome,passes=passes,raw=str(raw)),ensure_ascii=False))


if __name__ == '__main__':
    main()
