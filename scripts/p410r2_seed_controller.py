"""Measured original-task second-seed training, gated by successful stage A compute."""
from pathlib import Path
import json
import subprocess
import os
import time
import csv

ROOT=Path('/root/autodl-tmp/lerobot-main')
WORK=ROOT/'experiments/p4_public_benchmark/p4_10_gate_sweep/index3'
OUT=WORK/'result';PY='/root/autodl-tmp/p48_isolated_env/bin/python'
SCRIPT=WORK/'workspace/scripts/p410r2_seed_worker.py'
VARIANTS=['G0','G1e-2','G5e-2','G1e-1','NOOBJ']

def group(variants,mode,tag):
    processes=[]
    for v in variants:
        with (OUT/f'seed2_{mode}_{tag}_{v}.log').open('a') as log:
            p=subprocess.Popen([PY,'-u',str(SCRIPT),'--variant',v,'--mode',mode,'--tag',tag],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        processes.append(p)
    assert all(p.wait()==0 for p in processes),'See per-variant logs'

def main():
    import fcntl
    lock=(OUT/'seed2_controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='WAITING_A',pid=os.getpid(),seed=2,budget_each=5000,classification='REPLICATION_ON_KNOWN_STARTS')
    target=OUT/'known_seed_replication_status.json'
    target.write_text(json.dumps(state,indent=2))
    try:
        while True:
            a=json.loads((OUT/'status.json').read_text())
            if a['status']=='A_FAILED':raise RuntimeError('Stage A compute failed; no B launch')
            if a['status']=='A_COMPUTE_COMPLETE_REVIEW_PENDING' and (OUT/'fixed_alpha_evaluation_complete.json').exists():break
            time.sleep(15)
        assert json.loads((OUT/'semantic_review/replay_complete.json').read_text())['status']=='DONE'
        review=json.loads((OUT/'semantic_gate.json').read_text())
        assert review['allow_known_start_replication'] and not review['confirmed_prototype_failure']
        state['status']='SMOKE';target.write_text(json.dumps(state,indent=2))
        for v in VARIANTS:group([v],'smoke','single')
        records=[];single={};hashes={};chosen=1;prior=1.
        for n in [1,2,3,4]:
            groups=[[v] for v in VARIANTS[:4]] if n==1 else [VARIANTS[:n]]
            for vs in groups:
                tag='n'+str(n);group(vs,'bench',tag)
                reports=[]
                for v in vs:
                    r=json.loads((OUT/f'known_seed2/{v}/bench/fixed/{tag}/benchmark.json').read_text())
                    hashes.setdefault(v,r['final_model_sha256']);assert hashes[v]==r['final_model_sha256']
                    records.append(dict(variant=v,concurrency=n,**r));reports.append(r)
                    if n==1:single[v]=r['timed_wall_seconds']
                gain=sum(single[v] for v in vs)/max(r['timed_wall_seconds'] for r in reports)
                if n>1 and gain>=1.1*prior:chosen=n;prior=gain
        with (OUT/'known_seed2_throughput.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
        state.update(status='TRAINING',concurrency=chosen);target.write_text(json.dumps(state,indent=2))
        for i in range(0,len(VARIANTS),chosen):group(VARIANTS[i:i+chosen],'train','formal')
        state['status']='SEED2_5K_COMPLETE_EVALUATION_PENDING'
    except Exception as error:state.update(status='FAILED',error=str(error))
    target.write_text(json.dumps(state,indent=2))

if __name__=='__main__':main()
