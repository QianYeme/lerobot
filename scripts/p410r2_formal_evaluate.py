"""Same Dev15 evaluator for all four fresh20k pilots; honest phase-evidence limits."""
from pathlib import Path
import argparse
import csv
import json
import subprocess
import sys
import time

ROOT=Path('/root/autodl-tmp/lerobot-main');BASE=ROOT/'experiments/p4_public_benchmark/p4_10_gate_sweep'
WORK=BASE/'index3';OUT=WORK/'result';P49=ROOT/'experiments/p4_final_p1_training/p4_9_unified_gate_init/index1'
JOBS={'C_L1_G0':('f6','legacy',1,20000),'C_L1_G05':('f6','legacy',1,20000),
      'C_F3_G0':('f6','fixed',3,20000),'C_F3_G05':('f6','fixed',3,20000)}

def evaluate(job):
    sys.path.insert(0,str(P49/'workspace/scripts'))
    import torch
    import p49_runtime as runtime
    import p49_evaluate as evaluator
    runtime.JOBS.update(JOBS)
    evaluator.OUT=OUT/'formal6_eval'
    def load(_job,_historical):
        old,cfg,policy,opt,data,pre,stats=runtime.factory(job,evaluator.OUT/'assets'/job,.05 if job.endswith('G05') else 0.)
        del opt,data
        latest=json.loads((OUT/f'formal6_pilot/runs/{job}/latest.json').read_text())
        assert latest['step']==20000 and runtime.sha(latest['path'])==latest['sha256']
        state=torch.load(latest['path'],map_location='cpu',weights_only=False);assert state['step']==20000
        policy.load_state_dict(state['model'],strict=True);del state;policy.eval()
        return old,cfg,policy,pre,stats,latest['path']
    evaluator.load=load;evaluator.dev(job,False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',choices=list(JOBS));a=p.parse_args()
    if a.job:evaluate(a.job);return
    while True:
        state=json.loads((OUT/'formal6_pilot_status.json').read_text())
        if state['status']=='FAILED':raise RuntimeError(state)
        if state['status']=='FOUR_20K_PILOTS_COMPLETE_EVALUATION_PENDING':break
        time.sleep(15)
    for i in range(0,4,2):
        processes=[]
        for job in list(JOBS)[i:i+2]:
            with (OUT/f'{job}_dev.log').open('a') as log:
                processes.append(subprocess.Popen(['/root/miniconda3/bin/python','-u',__file__,'--job',job],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL))
        assert all(p.wait()==0 for p in processes),'See formal6 eval logs'
    rows=[];objects=[]
    for p in (OUT/'formal6_eval/evaluation').glob('*/dev_metrics.csv'):
        rows.extend(csv.DictReader(p.open()))
        objects.extend(json.loads((p.parent/'object_path.json').read_text()))
    for name,records in [('formal6_20k_comparison.csv',rows),('formal6_object_path_diagnostics.csv',objects)]:
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (OUT/'formal6_phase_evidence_limits.md').write_text('''# Stage analysis limits

The same historical6 fractions across all15 Dev episodes are scored, with valid chunk padding masks and h0 metrics separately. These fractions are progress proxies, not GT contact/lift/transport/place labels. No reviewed formal6 action-stage annotations were located in the frozen artifact catalog. Do not rename fractions as physical stages or infer force/contact correctness from them. Quantitative stage-specific selection remains unverified; report candidates as provisional/inconclusive rather than claiming a full stage gain.
''')
    (OUT/'formal6_pilot_evaluation_complete.json').write_text(json.dumps(dict(status='DONE_CORE_METRICS_PHASE_LABELS_UNVERIFIED',models=list(JOBS),final15_accessed=False)))

if __name__=='__main__':main()
