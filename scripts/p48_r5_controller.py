"""One SSH-independent r5 queue: gates -> measured scheduling -> three10k -> eval -> package."""
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from p48_r5_common import OUT,WORK,sha,write

PYTHON='/root/autodl-tmp/p48_isolated_env/bin/python'
SCRIPTS=WORK/'workspace/scripts'
KINDS=['act','legacy','fixed']


def run_group(stage,commands):
    processes=[];started=time.time()
    for name,command in commands:
        path=OUT/'queue_logs'/f'{stage}_{name}.log';path.parent.mkdir(exist_ok=True)
        log=path.open('a');process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        processes.append((name,process,log))
    write(OUT/'active_stage.json',dict(stage=stage,started=started,processes=[dict(name=n,pid=p.pid) for n,p,_ in processes]))
    failures=[]
    for name,process,log in processes:
        code=process.wait();log.close()
        if code:failures.append(dict(name=name,exit_code=code))
    if failures:raise RuntimeError(f'{stage} failed: {failures}; all logs retained')
    return time.time()-started


def worker(kind,mode,tag='single'):
    return [PYTHON,'-u',str(SCRIPTS/'p48_r5_worker.py'),'--kind',kind,'--mode',mode,'--tag',tag]


def benchmark():
    records=[];reports={}
    groups=[['act'],['legacy'],['fixed'],['legacy','fixed'],KINDS]
    for repeat in [1,2]:
        for group in groups:
            group_name='_'.join(group);tag=f'{group_name}_repeat{repeat}'
            done=all((OUT/'bench'/k/tag/'benchmark.json').exists() for k in group)
            if not done:run_group('bench_'+tag,[(k,worker(k,'bench',tag)) for k in group])
            for kind in group:
                r=json.loads((OUT/'bench'/kind/tag/'benchmark.json').read_text())
                reports[(repeat,group_name,kind)]=r
                records.append(dict(repeat=repeat,group=group_name,kind=kind,
                    updates_per_second=r['updates_per_second'],timed_updates=r['timed_updates'],
                    timed_wall_seconds=r['timed_wall_seconds'],peak_memory_bytes=r['peak_memory_bytes'],
                    final_model_sha256=r['final_model_sha256'],discarded_updates=True))
    # Same initial weights, sampler and100 updates must give the same state for each model
    # across single/parallel probes. This directly checks numerical resource isolation.
    for kind in KINDS:
        hashes={r['final_model_sha256'] for (_,_,k),r in reports.items() if k==kind}
        assert len(hashes)==1,(kind,'parallel changed effective updates',hashes)
    estimates=[]
    for repeat in [1,2]:
        single={k:reports[(repeat,k,k)]['updates_per_second'] for k in KINDS}
        serial=sum(10000/single[k] for k in KINDS)
        pair=10000/single['act']+max(10000/reports[(repeat,'legacy_fixed',k)]['updates_per_second'] for k in ['legacy','fixed'])
        triple=max(10000/reports[(repeat,'act_legacy_fixed',k)]['updates_per_second'] for k in KINDS)
        estimates.append(dict(repeat=repeat,serial_seconds=serial,pair_then_act_seconds=pair,triple_seconds=triple,
                              pair_speedup=serial/pair,triple_speedup=serial/triple))
    candidates={'serial':sum(x['serial_seconds'] for x in estimates)/2}
    if all(x['pair_speedup']>=1.10 for x in estimates):candidates['pair_then_act']=sum(x['pair_then_act_seconds'] for x in estimates)/2
    if all(x['triple_speedup']>=1.10 for x in estimates):candidates['triple']=sum(x['triple_seconds'] for x in estimates)/2
    chosen=min(candidates,key=candidates.get)
    with (OUT/'throughput_benchmark.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    result=dict(status='MEASURED_SCHEDULING_READY',chosen=chosen,estimates=estimates,
                eligible_total_time_seconds=candidates,parallel_next_state_identical=True,
                minimum_repeated_aggregate_speedup=1.10,formal_updates_per_model=10000)
    write(OUT/'scheduling_decision.json',result)
    return chosen


def main():
    lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ['LD_LIBRARY_PATH']='/root/autodl-tmp/p48_system_libs/usr/lib/x86_64-linux-gnu:'+os.environ.get('LD_LIBRARY_PATH','')
    os.environ['MUJOCO_GL']='egl';os.environ['PYOPENGL_PLATFORM']='egl'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    assert json.loads((OUT/'prototype_gate.json').read_text())['status']=='G2_PROTOTYPE_PASS'
    assert json.loads((OUT/'timing_contract.json').read_text())['status']=='TIMEBASE_PASS_NATIVE_STEP_PROTOCOL'
    state=dict(controller_pid=os.getpid(),status='RUNNING',started=time.time(),pilot_only=True)
    write(OUT/'status.json',state)
    try:
        smoke=[]
        for kind in KINDS:
            path=OUT/'smoke'/kind/'validation.json'
            if not path.exists():run_group('smoke_'+kind,[(kind,worker(kind,'smoke'))])
            smoke.append(json.loads(path.read_text()))
        assert all(s['next_update_max_difference']==0 for s in smoke)
        assert len({s['initial_common_sha256'] for s in smoke})==1
        assert smoke[1]['initial_model_sha256']==smoke[2]['initial_model_sha256']
        assert all(s['initial_cnn_inputs_and_features']==smoke[0]['initial_cnn_inputs_and_features'] for s in smoke)
        write(OUT/'paired_implementation_gate.json',dict(status='PAIRED_COMMON_INIT_CNN_AND_PREPROCESS_PASS',
              shared_common_initial_sha256=smoke[0]['initial_common_sha256'],
              paired_p1_initial_model_sha256=smoke[1]['initial_model_sha256'],
              cnn_input_and_features_bitwise_identical=True,next_update_replay_differences=[s['next_update_max_difference'] for s in smoke],
              prototype_sha256=sha(OUT/'prototype_bank.pt'),legacy_fixed_difference='online DINO preprocessing only'))
        chosen=benchmark();state['schedule']=chosen;state['status']='TRAINING';write(OUT/'status.json',state)
        if chosen=='triple':groups=[KINDS]
        elif chosen=='pair_then_act':groups=[['legacy','fixed'],['act']]
        else:groups=[[k] for k in KINDS]
        for group in groups:
            unfinished=[k for k in group if not (OUT/'runs'/k/'training_complete.json').exists()]
            if unfinished:run_group('train',[(k,worker(k,'train')) for k in unfinished])
        state['status']='TRAINING_DONE_EVALUATING';write(OUT/'status.json',state)
        run_group('eval_video_files',[('fetch',[PYTHON,'-u',str(SCRIPTS/'p48_r5_fetch_eval_videos.py')])])
        run_group('dev_cache',[('dev',[PYTHON,'-u',str(SCRIPTS/'p48_r5_cache.py'),'--split','dev'])])
        run_group('dev9',[(k,[PYTHON,'-u',str(SCRIPTS/'p48_r5_evaluate.py'),'--kind',k,'--mode','dev']) for k in KINDS])
        # Independent initialized environments; each process owns its policy and10 paired reset IDs.
        run_group('paired_rollouts',[(k,[PYTHON,'-u',str(SCRIPTS/'p48_r5_evaluate.py'),'--kind',k,'--mode','rollout']) for k in KINDS])
        state['status']='PILOT_COMPUTE_COMPLETE';state['finished']=time.time();write(OUT/'status.json',state)
        run_group('result',[('package',[PYTHON,'-u',str(SCRIPTS/'p48_r5_finalize.py')])])
    except Exception:
        (OUT/'controller_error.log').write_text(traceback.format_exc())
        state['status']='ENGINEERING_STOP_REVIEW_ERROR_LOG';state['finished']=time.time();write(OUT/'status.json',state)
        raise


if __name__=='__main__':main()
