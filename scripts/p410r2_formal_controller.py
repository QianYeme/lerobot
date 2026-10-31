"""Stage-C gate-frozen matched pilots, with measured concurrency and explicit review gate."""
from pathlib import Path
import csv
import json
import os
import subprocess
import time

ROOT=Path('/root/autodl-tmp/lerobot-main');WORK=ROOT/'experiments/p4_public_benchmark/p4_10_gate_sweep/index3'
OUT=WORK/'result';PY='/root/miniconda3/bin/python';SCRIPT=WORK/'workspace/scripts/p410r2_formal_worker.py'
JOBS=['C_L1_G0','C_L1_G05','C_F3_G0','C_F3_G05']

def group(jobs,mode,tag):
    processes=[]
    for job in jobs:
        with (OUT/f'{job}_{mode}_{tag}.log').open('a') as f:
            processes.append(subprocess.Popen([PY,'-u',str(SCRIPT),'--job',job,'--mode',mode,'--tag',tag],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL))
    codes=[p.wait() for p in processes];assert not any(codes),(jobs,codes)

def main():
    import fcntl
    lock=(OUT/'formal_controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dest=OUT/'formal6_pilot';dest.mkdir(exist_ok=True)
    state=dict(status='WAITING_A_AND_SEED2_EVIDENCE',pid=os.getpid(),target_each=20000,
        gate_candidates=[0.,.05],final80k='NOT_AUTHORIZED')
    def write(): (OUT/'formal6_pilot_status.json').write_text(json.dumps(state,indent=2))
    write()
    try:
        while not (OUT/'known_seed_replication_evaluation_complete.json').exists():
            if (OUT/'known_seed_replication_status.json').exists():
                previous=json.loads((OUT/'known_seed_replication_status.json').read_text())
                if previous['status']=='FAILED':raise RuntimeError('Seed2 branch failed; inspect before C')
            time.sleep(15)
        gate=json.loads((OUT/'semantic_gate.json').read_text())
        assert gate['allow_formal6_task_difference_pilot'] and not gate['confirmed_prototype_failure']
        (dest/'job_registry.json').write_text(json.dumps(dict(training_deadline=10**12,
            scope='fresh20k matched pilots only; no80k',jobs=JOBS)))
        state['status']='SMOKE';write()
        for job in JOBS:group([job],'smoke','single')
        # Paired non-gate tensor fingerprints must match within each architecture/seed.
        for pair in [JOBS[:2],JOBS[2:]]:
            hs=[json.loads((dest/f'smoke/{j}/identity.json').read_text())['initial_non_gate_sha256'] for j in pair]
            assert hs[0]==hs[1],pair
        records=[];single={};hashes={};chosen=1;gain_previous=1.
        for n in [1,2,3,4]:
            for jobs in ([[j] for j in JOBS] if n==1 else [JOBS[:n]]):
                tag='n'+str(n);group(jobs,'bench',tag);reports=[]
                for j in jobs:
                    r=json.loads((dest/f'bench/{j}/{tag}/benchmark.json').read_text());reports.append(r)
                    hashes.setdefault(j,r['final_sha256']);assert hashes[j]==r['final_sha256']
                    records.append(dict(concurrency=n,**r))
                    if n==1:single[j]=r['timed_seconds']
                gain=sum(single[j] for j in jobs)/max(r['timed_seconds'] for r in reports)
                if n>1 and gain>=1.1*gain_previous:chosen=n;gain_previous=gain
        with (OUT/'formal6_pilot_throughput.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
        state.update(status='TRAINING',concurrency=chosen);write()
        for i in range(0,len(JOBS),chosen):group(JOBS[i:i+chosen],'train','formal')
        state['status']='FOUR_20K_PILOTS_COMPLETE_EVALUATION_PENDING'
    except Exception as error:state.update(status='FAILED',error=str(error))
    write()

if __name__=='__main__':main()
