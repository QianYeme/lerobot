"""SSH-independent P4.9 gates -> measured parallel queue -> paired evaluation."""
import fcntl
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from p49_runtime import OUT, WORK, JOBS, row, write

PY_LB='/root/autodl-tmp/p48_isolated_env/bin/python'
PY_F6='/root/miniconda3/bin/python'
SCRIPTS=WORK/'workspace/scripts'

def start(job,script,args,label):
    env=dict(os.environ,HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONNOUSERSITE='1',
        LD_LIBRARY_PATH='/root/autodl-tmp/p48_system_libs/usr/lib/x86_64-linux-gnu:'+os.environ.get('LD_LIBRARY_PATH',''))
    path=OUT/'queue_logs'/f'{label}.log';path.parent.mkdir(exist_ok=True)
    log=path.open('a')
    command=[PY_F6 if JOBS[job][0]=='f6' else PY_LB,'-u',str(SCRIPTS/script),'--job',job]+args
    process=subprocess.Popen(command,env=env,cwd=WORK,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    log.close();return process

def wait_group(jobs,mode,tag):
    pending=[j for j in jobs if not (OUT/mode/j/(tag if mode=='bench' else '')/('benchmark.json' if mode=='bench' else 'validation.json')).exists()]
    processes={j:start(j,'p49_worker.py',['--mode',mode,'--tag',tag],mode+'_'+tag+'_'+j) for j in pending}
    while any(p.poll() is None for p in processes.values()):
        row(OUT/'resources.csv',dict(time=time.time(),stage=tag,
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),
            load=os.getloadavg()[0]))
        time.sleep(5)
    return {j:processes[j].returncode if j in processes else 0 for j in jobs}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume-readiness',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time()
    if (OUT/'job_registry.json').exists():
        assert args.resume_readiness,'Existing registry; explicit verified readiness resume required'
        registry=json.loads((OUT/'job_registry.json').read_text())
        assert registry['status'] in ['PREFLIGHT','THROUGHPUT_BENCHMARK']
        assert not list((OUT/'runs').glob('*/heartbeat.json')),'No readiness resume after formal training starts'
        registry.setdefault('controller_history',[]).append(registry['pid'])
        registry['pid']=os.getpid()
    else:
        registry=dict(status='PREFLIGHT',pid=os.getpid(),started=started,finish_deadline=started+10*3600,
            training_deadline=started+8.75*3600,jobs={j:dict(status='PENDING',target=v[3]) for j,v in JOBS.items()},failures=[])
    write(OUT/'job_registry.json',registry)
    def persist():write(OUT/'job_registry.json',registry)
    try:
        assert json.loads((OUT/'preflight_complete.json').read_text())['status']=='PASS'
        eligible=[]
        for job in JOBS:
            registry['jobs'][job]['status']='SMOKE';persist()
            result=wait_group([job],'smoke','identity')
            if result[job]==0:
                eligible.append(job);registry['jobs'][job]['status']='READY'
            else:
                registry['jobs'][job]['status']='BLOCKED'
                registry['failures'].append(dict(stage='smoke',job=job,log=f'queue_logs/smoke_identity_{job}.log'))
            persist()
        assert eligible,'No unblocked jobs'
        registry['status']='THROUGHPUT_BENCHMARK';persist()
        single={};final_hashes={};measured={};max_concurrency=1;previous_gain=1.
        groups=[[j] for j in eligible]+[eligible[:n] for n in range(2,len(eligible)+1)]
        for group in groups:
            tag='n'+str(len(group))+'_'+'_'.join(group)
            results=wait_group(group,'bench',tag)
            if any(results.values()):
                registry['failures'].append(dict(stage='benchmark',group=group,exit_codes=results));persist();continue
            reports={j:json.loads((OUT/'bench'/j/tag/'benchmark.json').read_text()) for j in group}
            hashes_valid=True
            for j,r in reports.items():
                final_hashes.setdefault(j,r['final_sha256'])
                hashes_valid &= final_hashes[j]==r['final_sha256']
                if len(group)==1:single[j]=r['updates_per_second']
                row(OUT/'throughput_benchmark.csv',dict(r,group=tag,concurrency=len(group)))
            if not hashes_valid:
                registry['failures'].append(dict(stage='parallel_numerical_identity',group=group));persist();continue
            if all(j in single for j in group):
                parallel=max(r['timed_seconds'] for r in reports.values())
                gain=sum(80/single[j] for j in group)/parallel
                measured[len(group)]=dict(serial_relative_speedup=gain,aggregate_updates_per_second=sum(r['updates_per_second'] for r in reports.values()),peak_bytes=sum(r['peak_memory_bytes'] for r in reports.values()))
                # Raise concurrency only when measured completion throughput adds >=10%.
                if len(group)>1 and gain>=1.1*previous_gain:
                    max_concurrency=len(group);previous_gain=gain
        write(OUT/'scheduling_decision.json',dict(max_concurrency=max_concurrency,measurements=measured,
            priority=list(JOBS),isolation='separate processes, samplers, logs; numerical final state checked',
            comparison='mixed-domain aggregate updates differ in cost; speedup uses sum serial times / parallel walltime'))
        registry['status']='TRAINING';registry['max_concurrency']=max_concurrency;persist()
        pending=list(eligible);active={};evaluation=[];eval_pending=[];eval_active={};eval_ok=True;act_scheduled=False
        registry['evaluation_completed']=[]
        def enqueue_eval(job):
            nonlocal act_scheduled
            for historical in [False,True]:
                suffix='_baseline' if historical else ''
                hargs=['--historical'] if historical else []
                eval_pending.append((job,['--mode','dev']+hargs,job+suffix+'_dev'))
                if JOBS[job][0]=='lb':eval_pending.append((job,['--mode','rollout']+hargs,job+suffix+'_rollout'))
            if JOBS[job][0]=='lb' and not act_scheduled:
                act_scheduled=True
                eval_pending.extend([(job,['--mode','act_dev'],'act_dev'),(job,['--mode','act_rollout'],'act_rollout')])
        while pending or active or eval_pending or eval_active:
            while pending and len(active)<max_concurrency and time.time()<registry['training_deadline']:
                job=pending.pop(0)
                process=start(job,'p49_worker.py',['--mode','train'],'train_'+job)
                active[job]=process;registry['jobs'][job].update(status='TRAINING',pid=process.pid);persist()
            for job,process in list(active.items()):
                if process.poll() is None:continue
                del active[job]
                complete=OUT/'runs'/job/'training_complete.json'
                result=json.loads(complete.read_text()) if complete.exists() else dict(status='FAILED')
                registry['jobs'][job].update(result)
                if process.returncode:registry['failures'].append(dict(stage='train',job=job,exit_code=process.returncode))
                if result['status']=='DONE':evaluation.append(job);enqueue_eval(job)
                persist()
            if pending and time.time()>=registry['training_deadline']:
                for job in pending:registry['jobs'][job]['status']='NOT_STARTED_BUDGET'
                pending=[];persist()
            # Completed runs are evaluated as soon as a measured slot is free.
            # New training jobs retain priority; at most two evaluators share the GPU.
            while eval_pending and not pending and len(active)+len(eval_active)<max_concurrency and len(eval_active)<2:
                job,arguments,label=eval_pending.pop(0)
                process=start(job,'p49_evaluate.py',arguments,label);eval_active[label]=process
            for label,process in list(eval_active.items()):
                if process.poll() is None:continue
                del eval_active[label]
                if process.returncode:
                    eval_ok=False;registry['failures'].append(dict(stage='evaluation',task=label,exit_code=process.returncode))
                else:registry['evaluation_completed'].append(label)
                persist()
            registry['active_evaluators']={label:p.pid for label,p in eval_active.items()}
            if not active and not pending:registry['status']='EVALUATING'
            persist()
            time.sleep(5)
        registry['all_evaluation_tasks_done']=eval_ok and len(evaluation)==4
        registry['status']='DONE' if registry['all_evaluation_tasks_done'] else 'PARTIAL_OR_BLOCKED';persist()
    except Exception:
        registry['status']='FAILED';registry['failures'].append(dict(stage='controller',traceback=traceback.format_exc()));persist()
    subprocess.run([PY_F6,str(SCRIPTS/'p49_finalize.py')],check=True)

if __name__=='__main__':main()
