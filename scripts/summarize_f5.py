"""Export manually reviewed F5 stages and runtime diagnostics; never invent success."""
import argparse
import csv
import json
from pathlib import Path

MODELS=['P1-Hist','Strong-G','L-Load','New-GL']
LABELS=['PASS','FAIL','UNKNOWN']


def write(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def summarize(session):
    stages=[];assistance=[];regulators=[];trials=[]
    manifest=[]
    if (session/'trial_manifest.csv').exists():
        with (session/'trial_manifest.csv').open(newline='',encoding='utf-8') as f:manifest=list(csv.DictReader(f))
    for path in sorted(session.glob('F5_*/summary.json')):
        runtime=json.loads(path.read_text());review=json.loads((path.parent/'stage_review.json').read_text())
        identity={k:runtime[k] for k in ['MODEL_ID','TRIAL_ID','POSITION']}
        if identity!={k:review[k] for k in identity}:raise ValueError('Review/trial identity mismatch')
        for phase in ['A','B']:
            labels=review[phase]
            if any(labels[f'S{i}'] not in LABELS for i in range(7)):raise ValueError('Stages must be PASS/FAIL/UNKNOWN')
            used=phase=='A' or runtime['assisted_recovery_used']
            full=labels['full_autonomous_success' if phase=='A' else 'full_chain_after_assistance']
            if full not in LABELS:raise ValueError('Full-chain label must be PASS/FAIL/UNKNOWN')
            start=0
            if phase=='B' and full=='PASS':
                failure=review['first_autonomous_failure_stage']
                if failure not in [f'S{i}' for i in range(7)]:raise ValueError('Assisted full-chain PASS needs first autonomous failure stage')
                start=max(2,int(failure[1]))
            if full=='PASS' and (not used or any(labels[f'S{i}']!='PASS' for i in range(start,7))):raise ValueError('Full-chain PASS needs observed PASS for every applicable stage')
            if phase=='A' and runtime['assisted_recovery_used'] and full!='FAIL':raise ValueError('assist command confirmed autonomous failure: A must remain FAIL')
            stages.append(dict(**identity,phase=phase,phase_used=used,validity=review['validity'],
                **{f'S{i}':labels[f'S{i}'] for i in range(7)},full_chain=full,
                visible_slip=review['visible_slip'],premature_release=review['premature_release'],visible_over_closure=review['visible_over_closure'],cup_deformation=review['cup_deformation'],
                operator_emergency_stop=review['operator_emergency_stop'],assisted_recovery_used=runtime['assisted_recovery_used'],
                approximate_lift_height_cm=review['approximate_lift_height_cm'],approximate_hold_duration_s=review['approximate_hold_duration_s']))
        if runtime['assisted_recovery_used']:
            assistance.append(dict(**identity,first_autonomous_failure_stage=review['first_autonomous_failure_stage'],assistance_type=review['assistance_type'],
                stages_recovered=','.join(f'S{i}' for i in range(7) if review['B'][f'S{i}']=='PASS'),full_chain_after_assistance=review['B']['full_chain_after_assistance']))
        metric_path=path.parent/'local_regulator_metrics.json'
        if metric_path.exists():
            metric=json.loads(metric_path.read_text())
            for phase,value in metric['by_phase'].items():regulators.append(dict(**identity,phase=phase,**value))
        trials.append(runtime)
    missing=[r for r in manifest if r['TRIAL_ID'] not in {t['TRIAL_ID'] for t in trials}]
    if not trials:return dict(status='INCOMPLETE_RUNTIME_EVIDENCE' if missing else 'NOT_RUN',attempted_trials=len(missing),incomplete_trial_ids=[r['TRIAL_ID'] for r in missing],F5_COMPLETE=False)
    write(session/'per_trial_stage_metrics.csv',stages);write(session/'assisted_recovery_summary.csv',assistance);write(session/'local_regulator_metrics.csv',regulators)
    models=[];autonomous=[]
    for model in MODELS:
        rows=[r for r in stages if r['MODEL_ID']==model and r['phase']=='A'];lost=sum(r['MODEL_ID']==model for r in missing);n=len(rows)+lost
        for stage in [f'S{i}' for i in range(7)]:
            models.append(dict(MODEL_ID=model,stage=stage,planned_trials=3,attempted_trials=n,pass_count=sum(r[stage]=='PASS' for r in rows) if n else None,
                fail_count=sum(r[stage]=='FAIL' for r in rows) if n else None,unknown_count=sum(r[stage]=='UNKNOWN' for r in rows)+lost if n else None))
        autonomous.append(dict(MODEL_ID=model,planned_trials=3,attempted_trials=n,autonomous_success_count=sum(r['full_chain']=='PASS' for r in rows) if n else None,
            unknown_count=sum(r['full_chain']=='UNKNOWN' for r in rows)+lost if n else None,interpretation='DESCRIPTIVE_PILOT_ONLY' if n else 'NOT_RUN'))
    write(session/'model_stage_summary.csv',models);write(session/'autonomous_success_summary.csv',autonomous)
    return dict(status='INCOMPLETE_RUNTIME_EVIDENCE' if missing else 'REVIEW_REQUIRED',attempted_trials=len(trials)+len(missing),planned_trials=12,incomplete_trial_ids=[r['TRIAL_ID'] for r in missing],F5_COMPLETE=False,
        reason='Scientific outcome and validity/phase evidence require executor/independent review; runtime termination does not imply task success')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--session-dir',type=Path,required=True);args=ap.parse_args()
    print(json.dumps(summarize(args.session_dir),indent=2))
