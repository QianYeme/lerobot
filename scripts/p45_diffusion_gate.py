"""Non-disruptive supplement: wait for the complete parent queue, benchmark only when idle."""
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

ROOT=Path('/root/autodl-tmp/lerobot-main')
PARENT=ROOT/'experiments/p4_final_p1_training/p4_5_robot_reliability_server10h/index1'
WORK=PARENT.parent/'supplements/diffusion_baseline/index1';OUT=WORK/'result'


def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)


def event(stage,status,reason,**extra):
    row=dict(time=time.time(),stage=stage,status=status,reason=reason,details=json.dumps(extra))
    path=OUT/'resource_gate_and_schedule.csv';exists=path.exists()
    with path.open('a',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(row))
        if not exists:writer.writeheader()
        writer.writerow(row)


def gpu_pids():
    value=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return {int(line.strip()) for line in value.splitlines() if line.strip().isdigit()}


def execute(run,mode,seed,deadline):
    env=os.environ.copy();env['PYTHONPATH']=str(PARENT/'stock/src')
    env.pop('CUDA_VISIBLE_DEVICES',None)
    with (OUT/(run+'.log')).open('a') as log:
        process=subprocess.Popen([sys.executable,str(ROOT/'scripts/p45_diffusion_worker.py'),'--run',run,'--mode',mode,
            '--seed',str(seed),'--deadline',str(deadline)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(OUT/'active_job.json',dict(pid=process.pid,run=run,mode=mode,deadline=deadline))
    event(mode,'STARTED','Parent complete, exclusive GPU capacity confirmed',pid=process.pid,run=run)
    stop_reason=None
    while process.poll() is None:
        others=gpu_pids()-{process.pid}
        if time.time()>=deadline or others or shutil.disk_usage(WORK).free<16*1024**3:
            stop_reason='deadline' if time.time()>=deadline else 'other GPU job detected' if others else 'disk reserve'
            os.killpg(process.pid,signal.SIGTERM)
            hard=json.loads((PARENT/'result/server/campaign_started.json').read_text())['hard_end']
            try:process.wait(timeout=max(.1,min(45,hard-time.time()-5)))
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            break
        time.sleep(2)
    event(mode,'EXIT','No parent process modified',run=run,returncode=process.returncode,stop_reason=stop_reason)
    return process.returncode


def finish(status,reason,**extra):
    done=extra.get('completed_seeds',[])
    write(OUT/'execution_complete.json',dict(status=status,terminal=True,reason=reason,finished=time.time(),parent_modified=False,
        parent_gpu_overlap=False,robot='DEFERRED_BY_USER',deferred_seeds=[seed for seed in [1,2,3] if seed not in done],**extra))
    with zipfile.ZipFile(WORK/'DIFFUSION_BASELINE_result.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in OUT.rglob('*'):
            if path.is_file() and path.suffix in ['.json','.md','.csv','.log']:
                archive.write(path,path.relative_to(WORK).as_posix())
        archive.write(WORK/'decision.md','decision.md')


def main():
    lock=(OUT/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'scheduler_identity.json').exists():raise RuntimeError('Already registered; no duplicate supplement scheduler')
    campaign=json.loads((PARENT/'result/server/campaign_started.json').read_text());hard=campaign['hard_end']
    write(OUT/'scheduler_identity.json',dict(pid=os.getpid(),parent_started=campaign['started'],hard_end=hard,started=time.time()))
    smoke=OUT/'CPU_READINESS/smoke.json'
    if not smoke.exists() or json.loads(smoke.read_text())['status']!='PASS':
        finish('BLOCKED_SCHEMA','CPU real-data smoke did not pass');return
    source=json.loads((PARENT/'result/server/stock_source_manifest.json').read_text())
    for entry in source['files']:
        path=PARENT/'stock'/entry['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            finish('BLOCKED_SCHEMA','Pinned official source changed; no parent files modified');return
    event('WAIT_PARENT','QUEUED','No GPU allocation while the parent has pending work',hard_end=hard)
    write(OUT/'execution_complete.json',dict(status='DEFERRED_RESOURCE_LIMIT',reason='Current GPU busy; conditional tail scheduler waiting',
        terminal=False,parent_modified=False,parent_gpu_overlap=False,hard_end=hard))
    while time.time()<hard-600:
        current=json.loads((PARENT/'result/server/campaign_started.json').read_text())
        if current['started']!=campaign['started']:
            finish('DEFERRED_RESOURCE_LIMIT','Parent campaign changed; cannot assume a new budget window');return
        ready=PARENT/'result/server/delivery_ready.json'
        if ready.exists():
            parent=json.loads(ready.read_text())
            if parent['server']!='SERVER_COMPLETE':
                finish('DEFERRED_RESOURCE_LIMIT','Parent incomplete; Diffusion cannot displace higher-priority work');return
            if not gpu_pids():break
        time.sleep(30)
    else:
        finish('DEFERRED_RESOURCE_LIMIT','No complete parent and spare capacity before minimum evaluation reserve');return
    # Confirm that idle capacity is sustained, not a transient gap between parent jobs.
    time.sleep(10)
    if gpu_pids():finish('DEFERRED_RESOURCE_LIMIT','GPU capacity no longer idle');return
    identity=json.loads((OUT/'CPU_READINESS/identity.json').read_text())
    native_budget=int(identity['native_default_budget']);parameters=int(json.loads(smoke.read_text())['parameters'])
    forecast_bytes=parameters*4*3*5+parameters*4+2*1024**3
    if shutil.disk_usage(WORK).free<16*1024**3+forecast_bytes:
        finish('DEFERRED_RESOURCE_LIMIT','Insufficient space for all full checkpoints plus16GiB reserve',forecast_bytes=forecast_bytes);return
    rc=execute('GPU_READINESS','gpu-benchmark',1,min(hard-300,time.time()+600))
    report=OUT/'GPU_READINESS/benchmark.json'
    if rc!=0 or not report.exists():finish('DIFFUSION_PARTIAL','GPU compatibility benchmark failed; no formal baseline started');return
    bench=json.loads(report.read_text())
    if bench['steps']!=50:finish('DEFERRED_RESOURCE_LIMIT','Full benchmark did not finish');return
    completed=[]
    for seed in [1,2,3]:
        required=native_budget/bench['steps_per_s']*1.25+90*bench['native_inference_s']*1.5+300
        remaining=hard-time.time()
        event('FULL_BUDGET_GATE','PASS' if remaining>=required else 'DEFERRED_RESOURCE_LIMIT',
            'Native100k budget; not80k step matched',seed=seed,required_s=required,remaining_s=remaining,
            steps_per_s=bench['steps_per_s'],forecast_bytes=forecast_bytes)
        if remaining<required or shutil.disk_usage(WORK).free<16*1024**3+forecast_bytes:
            finish('DIFFUSION_COMPLETE' if completed else 'DEFERRED_RESOURCE_LIMIT','No capacity for another complete native-budget seed',completed_seeds=completed);return
        run=f'VDP-S{seed}';folder=OUT/run
        if folder.exists():
            finish('DIFFUSION_PARTIAL','Existing formal run needs an explicit exact-resume audit; no fresh automatic retry',completed_seeds=completed);return
        rc=execute(run,'train',seed,hard-60)
        done=folder/'execution_complete.json'
        if rc!=0 or not done.exists() or json.loads(done.read_text())['status']!='DIFFUSION_COMPLETE':
            finish('DIFFUSION_PARTIAL','Formal run/evaluation incomplete; evidence and checkpoint retained',completed_seeds=completed);return
        completed.append(seed)
    finish('DIFFUSION_COMPLETE','All permitted native-budget seeds complete',completed_seeds=completed)


if __name__=='__main__':main()
