"""R3 full official-bank task0 development evaluation; reuse validated 5k weights."""
import argparse
import csv
import fcntl
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path('/root/autodl-tmp/lerobot-main');BASE=ROOT/'experiments/p4_public_benchmark/p4_10_gate_sweep'
WORK=BASE/'index4';OUT=WORK/'result';R5=ROOT/'experiments/p4_public_benchmark/p4_8_libero_pilot/index5'
sys.path.insert(0,str(R5/'workspace/scripts'))
import p48_r5_common as old
VARIANTS=['G0','G1e-2','G5e-2','G1e-1','NOOBJ']

def checkpoint(seed,variant):
    if variant=='GFIXED':return BASE/'index3/result/fixed_alpha/train/GFIXED/checkpoints/005000/resume.pt'
    if seed==2:return BASE/f'index3/result/known_seed2/{variant}/runs/{"act" if variant=="NOOBJ" else "fixed"}/checkpoints/005000/resume.pt'
    if variant in ['G0','NOOBJ']:return R5/f'result/runs/{"act" if variant=="NOOBJ" else "fixed"}/checkpoints/005000/resume.pt'
    return BASE/f'index1/result/train/{variant}/checkpoints/005000/resume.pt'

def evaluate(seed,variant):
    import torch
    import p48_r5_evaluate as evaluator
    dest=OUT/f'task0/seed{seed}/{variant}';dest.mkdir(parents=True,exist_ok=True)
    lock=(dest/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=checkpoint(seed,variant);source_sha=old.sha(source)
    if (dest/'complete.json').exists():
        assert json.loads((dest/'complete.json').read_text())['checkpoint_sha']==source_sha
        return
    def load(_kind):
        kind='act' if variant=='NOOBJ' else 'fixed';cfg,policy=old.make_policy(kind)
        state=torch.load(source,map_location='cpu',weights_only=False);assert state['step']==5000
        policy.load_state_dict(state['model'],strict=True);policy.eval()
        return cfg,policy,dict(path=str(source),sha256=source_sha,step=5000)
    contract=json.loads((R5/'result/timing_contract.json').read_text())
    contract.update(paired_initial_ids=list(range(50)),episode_seeds=[1]*50)
    old.write(dest/'timing_contract.json',contract);shutil.copyfile(R5/'result/train35_statistics.pt',dest/'train35_statistics.pt')
    evaluator.load=load;evaluator.OUT=dest
    source_code=inspect.getsource(evaluator.rollout).replace('assert len(rows)==10','assert len(rows)==50').replace('episodes=10','episodes=50').replace('/10,','/50,').replace('TEN_PAIRED','FIFTY_DEVELOPMENT_PAIRED')
    exec(compile(source_code,'frozen_r5_rollout_full_official_bank','exec'),evaluator.__dict__)
    evaluator.rollout('act' if variant=='NOOBJ' else 'fixed')
    old.write(dest/'complete.json',dict(status='DONE',seed=seed,variant=variant,checkpoint_sha=source_sha,
        classification='FULL_OFFICIAL_BANK_DEVELOPMENT_SELECTION_NOT_HOLDOUT',init_ids=list(range(50))))

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int);p.add_argument('--variant');a=p.parse_args()
    if a.variant:evaluate(a.seed,a.variant);return
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'task0_controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='WAITING_SEED2_5K_WEIGHTS',pid=os.getpid(),task=0,init_ids=list(range(50)),classification='DEVELOPMENT_MODEL_SELECTION',scope='10 learnable/NOOBJ combinations +1 fixed-alpha mechanism control')
    old.write(OUT/'task0_status.json',state)
    try:
        while True:
            path=BASE/'index3/result/known_seed_replication_status.json'
            previous=json.loads(path.read_text())
            if previous['status']=='FAILED':raise RuntimeError(previous)
            if previous['status']=='SEED2_5K_COMPLETE_EVALUATION_PENDING':break
            time.sleep(15)
        state['status']='FULL50_EVALUATING';old.write(OUT/'task0_status.json',state)
        jobs=[(seed,v) for seed in [1,2] for v in VARIANTS]+[(1,'GFIXED')]
        for i in range(0,len(jobs),3):
            processes=[]
            for seed,v in jobs[i:i+3]:
                with (OUT/f'task0_seed{seed}_{v}.log').open('a') as f:
                    processes.append(subprocess.Popen([sys.executable,'-u',__file__,'--seed',str(seed),'--variant',v],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL))
            assert not any(p.wait() for p in processes),'See task0 per-model logs'
        rows=[]
        for path in (OUT/'task0').rglob('libero_rollouts.csv'):
            parts=path.relative_to(OUT/'task0').parts;seed=int(parts[0][4:]);variant=parts[1]
            for r in csv.DictReader(path.open()):
                r.update(training_seed=seed,variant=variant,classification='DEVELOPMENT_MODEL_SELECTION',dedup_key=f'task0_seed{seed}_{variant}_init{r["init_id"]}')
                rows.append(r)
        with (OUT/'task0_all_inits_paired.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        summary=[]
        for seed,v in jobs:
            subset=[r for r in rows if r['training_seed']==seed and r['variant']==v]
            summary.append(dict(training_seed=seed,variant=v,n=len(subset),successes=sum(r['success'].lower()=='true' for r in subset),gate_mode='fixed' if v=='GFIXED' else 'learnable_or_noobj'))
        with (OUT/'task0_seed_summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
        state['status']='FULL50_DEVELOPMENT_EVALUATION_DONE_REVIEW_PENDING'
    except Exception as error:state.update(status='FAILED',error=str(error))
    old.write(OUT/'task0_status.json',state)

if __name__=='__main__':main()
