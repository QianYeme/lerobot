"""Uniform original-task replication evidence; never call these unseen starts."""
from pathlib import Path
import argparse
import csv
import json
import subprocess
import sys
import time

ROOT=Path('/root/autodl-tmp/lerobot-main');BASE=ROOT/'experiments/p4_public_benchmark/p4_10_gate_sweep'
WORK=BASE/'index3';OUT=WORK/'result';VARIANTS=['G0','G1e-2','G5e-2','G1e-1','NOOBJ']
PY='/root/autodl-tmp/p48_isolated_env/bin/python'

def evaluate(variant,mode):
    sys.path.insert(0,str(BASE/'index1/workspace/scripts'))
    import p410_evaluate as evaluator
    old=evaluator.old;original=old.make_policy;seed=old.seed
    def factory(kind):
        try:
            old.seed=lambda ignored=1:seed(2)
            return original(kind)
        finally:old.seed=seed
    old.make_policy=factory
    dest=OUT/'known_seed2_eval';dest.mkdir(exist_ok=True)
    old.write(dest/'init_contract.json',json.loads((BASE/'index1/result/init_contract.json').read_text()))
    evaluator.OUT=dest
    evaluator.checkpoint=lambda v:OUT/f'known_seed2/{v}/runs/{"act" if v=="NOOBJ" else "fixed"}/checkpoints/005000/resume.pt'
    sys.argv=[sys.argv[0],'--variant',variant,'--mode',mode];evaluator.main()

def main():
    p=argparse.ArgumentParser();p.add_argument('--variant');p.add_argument('--mode');a=p.parse_args()
    if a.variant:evaluate(a.variant,a.mode);return
    while True:
        state=json.loads((OUT/'known_seed_replication_status.json').read_text())
        if state['status']=='FAILED':raise RuntimeError(state)
        if state['status']=='SEED2_5K_COMPLETE_EVALUATION_PENDING':break
        time.sleep(15)
    hashes=[]
    for v in VARIANTS:
        kind='act' if v=='NOOBJ' else 'fixed'
        hashes.append(json.loads((OUT/f'known_seed2/{v}/smoke/{kind}/validation.json').read_text())['initial_common_sha256'])
    assert len(set(hashes))==1,'Seed2 common backbone initialization differs'
    for mode in ['dev','rollout']:
        for i in range(0,len(VARIANTS),3):
            processes=[]
            for v in VARIANTS[i:i+3]:
                with (OUT/f'seed2_eval_{mode}_{v}.log').open('a') as log:
                    processes.append(subprocess.Popen([PY,'-u',__file__,'--variant',v,'--mode',mode],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL))
            assert all(p.wait()==0 for p in processes),'See seed2 eval logs'
    rows=[]
    for seed,folder in [(1,BASE/'index1/result/rollouts'),(2,OUT/'known_seed2_eval/rollouts')]:
        for path in folder.rglob('libero_rollouts.csv'):
            variant=path.relative_to(folder).parts[0]
            if variant not in VARIANTS:continue
            for row in csv.DictReader(path.open()):row.update(seed=seed,variant=variant,classification='REPLICATION_ON_KNOWN_STARTS');rows.append(row)
    with (OUT/'known_start_seed_replication.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'known_seed_replication_evaluation_complete.json').write_text(json.dumps(dict(status='DONE',rows=len(rows),common_initialization_match=True,known_starts=list(range(30,50))),indent=2))

if __name__=='__main__':main()
