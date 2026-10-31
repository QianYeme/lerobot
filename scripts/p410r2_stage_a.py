"""Persistent stage-A runner; no automatic promotion into B/C/final80k."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/root/autodl-tmp/lerobot-main')
WORK=ROOT/'experiments/p4_public_benchmark/p4_10_gate_sweep/index3'
OUT=WORK/'result';PY='/root/autodl-tmp/p48_isolated_env/bin/python'
SCRIPTS=WORK/'workspace/scripts'

def run(name,args):
    with (OUT/(name+'.log')).open('a') as log:
        p=subprocess.Popen([PY,'-u']+args,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        (OUT/'active_stage.json').write_text(json.dumps(dict(stage=name,pid=p.pid,time=time.time())))
        assert p.wait()==0,'See '+name+'.log'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(OUT/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='A_RUNNING',pid=os.getpid(),started=time.time(),B='NOT_STARTED',C='NOT_STARTED',final80k='NOT_AUTHORIZED')
    (OUT/'status.json').write_text(json.dumps(state,indent=2))
    try:
        run('semantic_replay',[str(SCRIPTS/'p410r2_semantic.py')])
        run('fixed_alpha_smoke',[str(SCRIPTS/'p410r2_fixed.py'),'--variant','GFIXED','--mode','smoke'])
        run('fixed_alpha_train',[str(SCRIPTS/'p410r2_fixed.py'),'--variant','GFIXED','--mode','train'])
        state['status']='A_COMPUTE_COMPLETE_REVIEW_PENDING'
    except Exception as error:state.update(status='A_FAILED',error=str(error))
    (OUT/'status.json').write_text(json.dumps(state,indent=2))

if __name__=='__main__':main()
