"""Summarize operator-reviewed stages and runtime separately, including interrupted trials."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from lerobot.utils.r5_10_protocol import MODELS,STAGE_NAMES


def write(path,rows,fields):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def summarize(session):
    trials=[];latency=[];initial=[];stages=[];assisted=[]
    for summary in sorted(session.glob('*/summary.json')):
        run=summary.parent;state=json.loads(summary.read_text());config=json.loads((run/'configuration.json').read_text())
        identity={k:state[k] for k in ['MODEL_ID','TRIAL_ID','POSITION']};identity['attempt_directory']=run.name
        audit=json.loads((run/'initial_state.json').read_text()) if (run/'initial_state.json').exists() else {}
        review=json.loads((run/'stage_review.json').read_text())
        valid=audit.get('status')=='RECORDED_MANUAL_RESET' and review.get('validity')=='VALID'
        source=run/(state['TRIAL_ID']+'.csv');rows=list(csv.DictReader(source.open(encoding='utf-8'))) if source.exists() else []
        values=np.array([float(r['inference_s']) for r in rows]);overruns=[str(r['overrun']).lower() in ['true','1'] for r in rows]
        latency.append(dict(**identity,samples=len(values),mean_inference_s=float(values.mean()) if len(values) else None,
            p95_inference_s=float(np.percentile(values,95)) if len(values) else None,max_inference_s=float(values.max()) if len(values) else None,
            control_overrun_fraction=float(np.mean(overruns)) if overruns else None,termination=state['status'],runtime_error=state.get('error'),
            safety_abort=bool(review.get('operator_emergency_stop'))))
        initial.append(dict(**identity,status=audit.get('status','MISSING_INITIAL_AUDIT'),state=json.dumps(audit.get('state',{})),
            load=audit.get('load'),valid_initial=audit.get('valid'),human_validity=review.get('validity'),valid_for_comparison=valid))
        assisted_used=state['assisted_recovery_used'];outcomes=review['A']
        full=outcomes['full_autonomous_success']
        if assisted_used and full=='PASS':raise ValueError('Assisted recovery cannot count as autonomous success: '+run.name)
        if full=='PASS' and not all(outcomes[k]=='PASS' for k in STAGE_NAMES):raise ValueError('Autonomous PASS requires all seven observed stages')
        stages.append(dict(**identity,**{k:outcomes[k] for k in STAGE_NAMES},full_autonomous_success=full,assisted_recovery=assisted_used,valid=valid))
        assisted.append(dict(**identity,used=assisted_used,first_autonomous_failure_stage=review.get('first_autonomous_failure_stage',''),
            assistance_type=review.get('assistance_type',''),phaseB_completion=review['B']['full_chain_after_assistance']))
        trials.append(dict(**identity,trial_kind=config['common'].get('trial_kind','primary'),termination=state['status'],rows=state['rows'],
            valid_for_comparison=valid,archived_attempt='_prev_' in run.name))
    for name,data,fields in [('trial_manifest',trials,['trial_kind','termination','rows','valid_for_comparison','archived_attempt']),
        ('initial_state_audit',initial,['status','state','load','valid_initial','human_validity','valid_for_comparison']),
        ('stage_outcomes',stages,list(STAGE_NAMES)+['full_autonomous_success','assisted_recovery','valid']),
        ('runtime_latency',latency,['samples','mean_inference_s','p95_inference_s','max_inference_s','control_overrun_fraction','termination','runtime_error','safety_abort']),
        ('assisted_recovery_log',assisted,['used','first_autonomous_failure_stage','assistance_type','phaseB_completion'])]:
        write(session/(name+'_reviewed.csv'),data,['MODEL_ID','TRIAL_ID','POSITION','attempt_directory']+fields)
    counts=[]
    for model in MODELS:
        actual=[r for r in stages if r['MODEL_ID']==model and r['valid']]
        counts.append(dict(model=model,valid_attempts=len(actual),**{k:sum(r[k]=='PASS' for r in actual) for k in STAGE_NAMES},
            autonomous_completions=sum(r['full_autonomous_success']=='PASS' for r in actual),assisted_recoveries=sum(r['assisted_recovery'] for r in actual)))
    write(session/'descriptive_counts.csv',counts,['model','valid_attempts']+list(STAGE_NAMES)+['autonomous_completions','assisted_recoveries'])
    return dict(recorded_attempts=len(trials),status='REVIEW_REQUIRED' if trials else 'NOT_RUN',R5_10_COMPLETE=False,
        note='Archived attempts kept; video stages manually reviewed; runtime/video not forced into timestamp alignment; no population success-rate claim')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--session-dir',type=Path,required=True);a=p.parse_args()
    print(json.dumps(summarize(a.session_dir),indent=2))
