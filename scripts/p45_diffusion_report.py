"""Supplement-only final aggregation and independent deadline fallback; no parent mutation."""
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import time
import zipfile

ROOT=Path('/root/autodl-tmp/lerobot-main')
PARENT=ROOT/'experiments/p4_final_p1_training/p4_5_robot_reliability_server10h/index1'
WORK=PARENT.parent/'supplements/diffusion_baseline/index1';OUT=WORK/'result'


def alive(pid,needle):
    try:return needle in Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:return False


def main():
    scheduler=json.loads((OUT/'scheduler_identity.json').read_text());pid=scheduler['pid'];hard=scheduler['hard_end']
    while time.time()<hard-5:
        job_path=OUT/'active_job.json'
        job=json.loads(job_path.read_text()) if job_path.exists() else None
        worker_alive=bool(job and alive(job['pid'],('--run\0'+job['run']+'\0').encode()) and alive(job['pid'],b'p45_diffusion_worker.py'))
        if not alive(pid,b'p45_diffusion_gate.py') and not worker_alive:break
        time.sleep(min(15,max(0,hard-5-time.time())))
    active=OUT/'active_job.json'
    if time.time()>=hard-5 and active.exists():
        job=json.loads(active.read_text());worker=job['pid'];needle=('--run\0'+job['run']+'\0').encode()
        if alive(worker,needle) and alive(worker,b'p45_diffusion_worker.py'):
            try:os.killpg(worker,signal.SIGKILL)
            except ProcessLookupError:pass
            (OUT/'deadline_fallback.json').write_text(json.dumps(dict(pid=worker,run=job['run'],time=time.time(),reason='Original parent hard deadline, own supplement group only'))+'\n')
    # Gate normally exits after writing its first ZIP. Never race that writer.
    while alive(pid,b'p45_diffusion_gate.py') and time.time()<hard+30:time.sleep(2)
    status_path=OUT/'execution_complete.json';status=json.loads(status_path.read_text())
    if not status.get('terminal',False):
        status.update(status='DIFFUSION_PARTIAL' if (OUT/'VDP-S1').exists() else 'DEFERRED_RESOURCE_LIMIT',terminal=True,
            reason='Scheduler ended or original hard deadline reached; no baseline completion inferred',finished=time.time())
        status_path.write_text(json.dumps(status,indent=2)+'\n')
    path=OUT/'baseline_comparison.md'
    text=path.read_text()+'\n\n## Terminal supplement result\n\n'+json.dumps(status,indent=2)+'\n'
    table=OUT/'baseline_h0_comparison.csv'
    if table.exists():
        with table.open() as stream:rows=list(csv.DictReader(stream))
        text+='\nSame Dev15 timepoints, physical h0 errors; no real-robot conclusion:\n\n|Run|Samples|Arm h0 MAE|Gripper h0 MAE|\n|---|---:|---:|---:|\n'
        for run in sorted({row['run'] for row in rows}):
            subset=[row for row in rows if row['run']==run]
            text+=f"|{run}|{len(subset)}|{statistics.mean(float(row['arm_h0_raw']) for row in subset):.6f}|{statistics.mean(float(row['gripper_h0_raw']) for row in subset):.6f}|\n"
    else:text+='\nNo full-budget Diffusion comparison exists; CPU/GPU smoke results are not benchmark performance.\n'
    legacy=PARENT/'result/server/legacy_checkpoint_per_sample.csv';normal=OUT/'normalization_contract.json'
    if legacy.exists() and normal.exists():
        std=json.loads(normal.read_text())['action_std'][5]
        with legacy.open() as stream:rows=list(csv.DictReader(stream))
        text+='\nParent Legacy P1 physical h0 gripper MAE (converted from its cached normalized h0 error with shared Train90 action std;8D observations, method-level only):\n\n'
        for step in sorted({row['step'] for row in rows},key=int):
            subset=[row for row in rows if row['step']==step]
            text+=f"- P1 {step}: {statistics.mean(float(row['h0_gripper_normalized']) for row in subset)*std:.6f}, {len(subset)} samples.\n"
        text+='\nParent horizon-averaged arm errors are not used as h0 arm errors.\n'
    path.write_text(text)
    files=[p for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.json','.md','.csv','.log','.txt'] and p.name!='delivery_manifest.json']
    manifest=[]
    for p in files:
        manifest.append(dict(path=p.relative_to(WORK).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    mp=OUT/'delivery_manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(WORK/'DIFFUSION_BASELINE_result.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in files+[mp,WORK/'decision.md']:archive.write(p,p.relative_to(WORK).as_posix())
    with zipfile.ZipFile(WORK/'DIFFUSION_BASELINE_result.zip') as archive:assert archive.testzip() is None
    (OUT/'delivery_ready.json').write_text(json.dumps(dict(status='READY',baseline_status=status['status'],zip=str(WORK/'DIFFUSION_BASELINE_result.zip')))+'\n')


if __name__=='__main__':main()
