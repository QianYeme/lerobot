"""Summarize human-reviewed R5 stages; never infer success from termination."""
import argparse
import csv
import json
from pathlib import Path

MODELS=['M1','M2','M3','M4','M5']
STAGES=[f'P{i}' for i in range(7)]
LABELS={'PASS','FAIL','UNKNOWN'}
VALID={'VALID','TASK_FAILURE'}


def write(path,records,fields):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(records)


def summarize(session):
    trials=[];recoveries=[];locals=[]
    for path in sorted(session.glob('R5_*/summary.json')):
        if '_prev_' in path.parent.name:continue
        runtime=json.loads(path.read_text());review=json.loads((path.parent/'stage_review.json').read_text())
        identity={key:runtime[key] for key in ['MODEL_ID','TRIAL_ID','POSITION']}
        if identity!={key:review[key] for key in identity}:raise ValueError('Trial/review identity mismatch')
        if identity['MODEL_ID'] not in MODELS:raise ValueError('Unexpected model')
        autonomous=review['A'];full=autonomous['full_autonomous_success']
        if full not in LABELS or any(autonomous[stage] not in LABELS for stage in STAGES):raise ValueError('Use PASS/FAIL/UNKNOWN')
        if full=='PASS' and (runtime['assisted_recovery_used'] or any(autonomous[s]!='PASS' for s in STAGES)):
            raise ValueError('Autonomous PASS requires all stages without assistance')
        if runtime['assisted_recovery_used'] and full!='FAIL':raise ValueError('Assistance marked autonomous failure; cannot relabel as autonomous success')
        trials.append(dict(**identity,validity=review['validity'],**{s:autonomous[s] for s in STAGES},
            full_autonomous_success=full,assisted_recovery_used=runtime['assisted_recovery_used'],termination=runtime['status']))
        if runtime['assisted_recovery_used']:
            recoveries.append(dict(**identity,first_autonomous_failure_stage=review['first_autonomous_failure_stage'],
                assistance_type=review['assistance_type'],recovered_stages=','.join(s for s in STAGES if review['B'][s]=='PASS'),
                full_chain_after_assistance=review['B']['full_chain_after_assistance']))
        local=path.parent/'local_regulator_metrics.json'
        if local.exists():
            values=json.loads(local.read_text())
            for phase,metric in values['by_phase'].items():
                locals.append(dict(**identity,phase=phase,**{k:metric[k] for k in ['samples','mean_abs','max_abs','near_bound_fraction','sign_change_rate','nonzero_fraction']}))
    common=['MODEL_ID','TRIAL_ID','POSITION']
    write(session/'per_trial_stage_metrics.csv',trials,common+['validity']+STAGES+['full_autonomous_success','assisted_recovery_used','termination'])
    write(session/'assisted_recovery_summary.csv',recoveries,common+['first_autonomous_failure_stage','assistance_type','recovered_stages','full_chain_after_assistance'])
    write(session/'local_regulator_metrics.csv',locals,common+['phase','samples','mean_abs','max_abs','near_bound_fraction','sign_change_rate','nonzero_fraction'])
    position=[];success=[]
    for model in MODELS:
        actual=[r for r in trials if r['MODEL_ID']==model];valid=[r for r in actual if r['validity'] in VALID]
        for stage in STAGES:
            position.append(dict(MODEL_ID=model,stage=stage,planned_trials=3,recorded_trials=len(actual),valid_trials=len(valid),
                pass_count=sum(r[stage]=='PASS' for r in valid),fail_count=sum(r[stage]=='FAIL' for r in valid),
                unknown_in_valid=sum(r[stage]=='UNKNOWN' for r in valid),unreviewed_or_invalid=len(actual)-len(valid),
                unrecorded_planned=max(0,3-len(actual))))
        success.append(dict(MODEL_ID=model,planned_trials=3,valid_trials=len(valid),autonomous_success_count=sum(r['full_autonomous_success']=='PASS' for r in valid),
            unknown_in_valid=sum(r['full_autonomous_success']=='UNKNOWN' for r in valid),unreviewed_or_invalid=len(actual)-len(valid),unrecorded_planned=max(0,3-len(actual))))
    write(session/'positioning_summary.csv',position,list(position[0]));write(session/'autonomous_success_summary.csv',success,list(success[0]))
    return dict(status='REVIEW_REQUIRED' if trials else 'NOT_RUN',recorded_trials=len(trials),planned_trials=15,R5_COMPLETE=False,
        note='Unrecorded/unreviewed/invalid observations are explicit; do not interpret zero observed PASS as zero success rate')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--session-dir',type=Path,required=True);args=parser.parse_args()
    args.session_dir.mkdir(parents=True,exist_ok=True);print(json.dumps(summarize(args.session_dir),indent=2))
