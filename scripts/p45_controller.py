"""Persistent, single-controller P4.5 ten-hour queue; never opens robot hardware."""
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile
import psutil

ROOT=Path('/root/autodl-tmp/lerobot-main')
WORK=ROOT/'experiments/p4_final_p1_training/p4_5_robot_reliability_server10h/index1'
OUT=WORK/'result/server'


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def row(path,obj):
    exists=path.exists()
    with path.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(obj))
        if not exists:w.writeheader()
        w.writerow(obj)


def launch(name,seed,deadline,benchmark=0,script='p45_vanilla_worker.py',stock=True,extra=()):
    env=os.environ.copy();env['PYTHONPATH']=str(WORK/'stock/src') if stock else str(ROOT/'src')
    log=(OUT/'training_logs');log.mkdir(exist_ok=True)
    stream=(log/(name+'.log')).open('a')
    cmd=[sys.executable,str(ROOT/'scripts'/script),'--run',name,'--seed',str(seed),'--deadline',str(deadline)]
    if benchmark:cmd+=['--benchmark',str(benchmark)]
    cmd+=list(extra)
    p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    stream.close();row(OUT/'schedule_actual.csv',dict(event='START',run=name,pid=p.pid,time=time.time(),returncode=''))
    return p


def wait_group(jobs,deadline):
    remaining=dict(jobs);results={};stopping=set()
    hard_end=json.loads((OUT/'campaign_started.json').read_text())['hard_end']
    last_resource=0
    while remaining:
        if shutil.disk_usage(WORK).free<16*1024**3 or time.time()>=deadline:
            for name,p in remaining.items():
                if name not in stopping:
                    os.killpg(p.pid,signal.SIGTERM);stopping.add(name)
        if time.time()>=min(deadline+300,hard_end):
            for p in remaining.values():
                if p.poll() is None:os.killpg(p.pid,signal.SIGKILL)
        if time.time()-last_resource>=10:
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
            row(OUT/'resources.csv',dict(time=time.time(),active=len(remaining),cpu_percent=psutil.cpu_percent(),
                ram_used=psutil.virtual_memory().used,ram_available=psutil.virtual_memory().available,gpu=gpu,
                free_disk=shutil.disk_usage(WORK).free))
            last_resource=time.time()
        for name,p in list(remaining.items()):
            rc=p.poll()
            if rc is not None:
                results[name]=rc;remaining.pop(name)
                row(OUT/'schedule_actual.csv',dict(event='EXIT',run=name,pid=p.pid,time=time.time(),returncode=rc))
        write(OUT/'controller_status.json',dict(pid=os.getpid(),active={k:v.pid for k,v in remaining.items()},updated=time.time()))
        if remaining:time.sleep(2)
    return results


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'campaign_started.json').exists():
        raise RuntimeError('Campaign already registered; inspect exact resume before restarting controller')
    started=time.time();train_end=started+8.75*3600;hard_end=started+10*3600
    write(OUT/'campaign_started.json',dict(started=started,training_cutoff=train_end,hard_end=hard_end,pid=os.getpid()))
    # Independent watchdog survives a crash in this controller; SIGTERM requests atomic checkpoint.
    watchdog=subprocess.Popen([sys.executable,'-c',
        "import time,os,signal,pathlib,json; p=pathlib.Path("+repr(str(OUT))+" ); time.sleep(max(0,"+repr(train_end)+"-time.time())); "
        "s=json.loads((p/'controller_status.json').read_text()); "
        "[(os.killpg(pid,signal.SIGTERM)) for pid in s.get('active',{}).values() if pathlib.Path('/proc/'+str(pid)+'/cmdline').exists() and b'p45_' in pathlib.Path('/proc/'+str(pid)+'/cmdline').read_bytes()]"],
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    write(OUT/'watchdog.json',dict(pid=watchdog.pid,cutoff=train_end))
    for count in (1,2,3):
        begin=time.time()
        jobs={f'BENCH_{count}_{i}':launch(f'BENCH_{count}_{i}',1,train_end,benchmark=200) for i in range(count)}
        codes=wait_group(jobs,train_end);elapsed=time.time()-begin
        valid=all(code==0 for code in codes.values())
        stats=[json.loads((OUT/'runs'/name/'benchmark.json').read_text()) for name in jobs] if valid else []
        row(OUT/'parallel_benchmark.csv',dict(workers=count,global_effective_steps_per_s=count*200/elapsed if valid else 0,
            walltime_s=elapsed,minimum_per_run_steps_per_s=min((s['steps_per_s'] for s in stats),default=0),
            max_per_run_peak_vram=max((s['peak_vram'] for s in stats),default=0),status='PASS' if valid else 'FAILED'))
    with (OUT/'parallel_benchmark.csv').open() as f:bench=list(csv.DictReader(f))
    chosen=1;throughput=float(bench[0]['global_effective_steps_per_s'])
    assert throughput>0,'Single-worker benchmark failed; no formal launch'
    for entry in bench[1:]:
        speed=float(entry['global_effective_steps_per_s'])
        if speed>=1.1*throughput and entry['status']=='PASS':chosen=int(entry['workers']);throughput=speed
    write(OUT/'parallel_decision.json',dict(workers=chosen,throughput=throughput,threshold='each additional concurrency level must gain >=10%'))
    pending=[('VACT-S1',1),('VACT-S2',2),('VACT-S3',3)]
    while pending and time.time()<train_end:
        group=pending[:chosen];pending=pending[chosen:]
        jobs={name:launch(name,seed,train_end) for name,seed in group};wait_group(jobs,train_end)
    # The NG worker owns its paired Global->Local DAG and can decline a scientifically invalid reconstruction.
    ng=ROOT/'scripts/p45_newgl_worker.py';contract=OUT/'newgl_schema_contract.json'
    if ng.exists() and contract.exists() and json.loads(contract.read_text()).get('status')=='PASS' and time.time()<train_end:
        ngjobs={name:launch(name,1,train_end,script=ng.name,stock=False,extra=['--stage','global']) for name in ['NG-L-G','NG-F-G']}
        result=wait_group(ngjobs,train_end)
        complete=all((OUT/'runs'/name/'execution_complete.json').exists() and
            json.loads((OUT/'runs'/name/'execution_complete.json').read_text())['status']=='DONE' for name in ngjobs)
        if complete and time.time()<train_end:
            local={name:launch(name,1,train_end,script=ng.name,stock=False,extra=['--stage','local']) for name in ['NG-L-L','NG-F-L']}
            wait_group(local,train_end)
    else:
        write(OUT/'newgl_execution.json',dict(status='BLOCKED',reason='No passing schema/preprocessing gate or remaining time'))
    evaluator=ROOT/'scripts/p45_legacy_eval.py'
    if evaluator.exists() and time.time()<hard_end-300:
        jobs={'LEGACY160-EVAL':launch('LEGACY160-EVAL',1,hard_end-120,script=evaluator.name,stock=False)}
        wait_group(jobs,hard_end-120)
    # Always package evidence; absence of real-robot data can never turn into success.
    runs={}
    for name in ['VACT-S1','VACT-S2','VACT-S3','NG-L-G','NG-F-G','NG-L-L','NG-F-L']:
        path=OUT/'runs'/name/'execution_complete.json'
        runs[name]=json.loads(path.read_text()) if path.exists() else dict(status='NOT_COMPLETED')
    status='SERVER_COMPLETE' if all(v['status']=='DONE' for v in runs.values()) and (OUT/'legacy_checkpoint_comparison.csv').exists() else 'SERVER_PARTIAL'
    write(WORK/'result/execution_complete.json',dict(server=status,robot='ROBOT_BLOCKED_OPERATOR_REQUIRED',runs=runs,
        finished=time.time(),elapsed_s=time.time()-started,final15_accessed=False))
    (OUT/'server_summary.md').write_text('# P4.5 server status\n\n'+status+'\n\n'+json.dumps(runs,indent=2)+'\n\nTeacher-forced Dev15 evidence is not robot reliability.\n')
    with zipfile.ZipFile(WORK/'P4_5_server_result.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in (WORK/'result').rglob('*'):
            if p.is_file() and p.suffix in ['.json','.jsonl','.csv','.md','.log'] and 'checkpoints' not in p.parts:
                archive.write(p,p.relative_to(WORK).as_posix())


if __name__=='__main__':main()
