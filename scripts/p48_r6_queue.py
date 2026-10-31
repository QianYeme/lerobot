"""SSH-independent, diagnostics-only queue, isolated from r5 outputs."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

ROOT=Path('/root/autodl-tmp/lerobot-main')
WORK=ROOT/'experiments/p4_public_benchmark/p4_8_libero_pilot/index6'
OUT=WORK/'result'
PYTHON='/root/autodl-tmp/p48_isolated_env/bin/python'
SCRIPTS=WORK/'workspace'


def write(value):
    temp=OUT/'status.partial';temp.write_text(json.dumps(value,indent=2));temp.replace(OUT/'status.json')


def group(tag,commands):
    started=time.time();jobs=[]
    for kind,command in commands:
        log=(OUT/f'{tag}_{kind}.log').open('a')
        p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        jobs.append((p,log))
    failures=[]
    for p,log in jobs:
        code=p.wait();log.close()
        if code:failures.append(code)
    if failures:raise RuntimeError(f'{tag} failed: {failures}; error logs preserved')
    return time.time()-started


def main():
    lock=(OUT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    state=dict(status='DIAGNOSTICS_RUNNING',pid=os.getpid(),started=time.time(),training_updates=0,new_rollouts=0)
    write(state)
    try:
        group('prepare',[('assets',[PYTHON,'-u',str(SCRIPTS/'p48_r6_prepare.py')])])
        def command(kind,pilot=False):
            return [PYTHON,'-u',str(SCRIPTS/'p48_r6_probe.py'),'--kind',kind]+(['--pilot'] if pilot else [])
        serial=0.
        for kind in ['legacy','fixed']:
            serial+=group('pilot_solo',[(kind,command(kind,True))])
            shutil.copyfile(OUT/f'pilot_{kind}.json',OUT/f'pilot_solo_{kind}.json')
        parallel=group('pilot_pair',[(k,command(k,True)) for k in ['legacy','fixed']])
        for kind in ['legacy','fixed']:shutil.copyfile(OUT/f'pilot_{kind}.json',OUT/f'pilot_pair_{kind}.json')
        use_pair=serial/parallel>=1.10
        schedule=dict(serial_end_to_end_seconds=serial,parallel_end_to_end_seconds=parallel,speedup=serial/parallel,
                      selected='parallel' if use_pair else 'serial',scope='loading+20 identical forward probes per model, not a policy performance comparison')
        (OUT/'diagnostic_schedule.json').write_text(json.dumps(schedule,indent=2))
        if use_pair:group('probes',[(k,command(k)) for k in ['legacy','fixed']])
        else:
            for kind in ['legacy','fixed']:group('probes',[(kind,command(kind))])
        state['status']='D0_D4_COMPUTE_DONE';state['finished']=time.time();write(state)
    except Exception:
        (OUT/'queue_error.log').write_text(traceback.format_exc());state['status']='ENGINEERING_ERROR';write(state);raise


if __name__=='__main__':main()
