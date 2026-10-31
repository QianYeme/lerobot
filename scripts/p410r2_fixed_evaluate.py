"""Complete fixed-vs-learnable A2 evaluation on the frozen known-start cohort."""
from pathlib import Path
import sys
import json
import time
import csv

ROOT=Path('/root/autodl-tmp/lerobot-main')
BASE=ROOT/'experiments/p4_public_benchmark/p4_10_gate_sweep'
WORK=BASE/'index3';OUT=WORK/'result'
sys.path.insert(0,str(BASE/'index1/workspace/scripts'))
import p410_evaluate as evaluation

def main():
    while True:
        state=json.loads((OUT/'status.json').read_text())
        if state['status']=='A_FAILED':raise RuntimeError('A failed')
        if state['status']=='A_COMPUTE_COMPLETE_REVIEW_PENDING':break
        time.sleep(10)
    folder=OUT/'fixed_alpha_eval';folder.mkdir(exist_ok=True)
    evaluation.OUT=folder
    evaluation.old.write(folder/'init_contract.json',json.loads((BASE/'index1/result/init_contract.json').read_text()))
    evaluation.VALUES['GFIXED']=.05
    evaluation.checkpoint=lambda _variant:OUT/'fixed_alpha/train/GFIXED/checkpoints/005000/resume.pt'
    original=evaluation.load
    def load(variant):
        cfg,policy,receipt=original(variant);policy.model.object_conditioning.gate.requires_grad_(False)
        return cfg,policy,receipt
    evaluation.load=load
    for mode in ['dev','rollout']:
        sys.argv=[sys.argv[0],'--variant','GFIXED','--mode',mode];evaluation.main()
    rows=[]
    for label,root in [('fixed_alpha',folder),('learnable_init',BASE/'index1/result')]:
        for p in (root/'evaluation').rglob('dev_metrics.csv'):
            if label=='learnable_init' and 'G5e-2' not in p.parts:continue
            for row in csv.DictReader(p.open()):row['alpha_mode']=label;rows.append(row)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'fixed_vs_learnable_gate.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    evaluation.old.write(OUT/'fixed_alpha_evaluation_complete.json',dict(status='DONE',known_init_ids=list(range(30,50)),confirmation=False))

if __name__=='__main__':main()
