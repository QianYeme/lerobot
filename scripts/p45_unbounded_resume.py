"""User-authorized completion queue, without a wall-clock budget; no robot I/O."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile

import p45_finalize as final

ROOT, WORK, OUT = final.ROOT, final.WORK, final.OUT
REC = WORK/'result/unbounded_resume'
DIFF = WORK.parent/'supplements/diffusion_baseline/index1/result'
# Existing workers require a numeric deadline. This is an API sentinel, not a budget.
NO_DEADLINE = '253402300799'


def write(name, value):
    path = REC/name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def execute(jobs):
    running = []
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'Foreign GPU work; stop rather than interfere'
    for run, script, extra, stock in jobs:
        env = os.environ.copy()
        env['PYTHONPATH'] = str(WORK/'stock/src' if stock else ROOT/'src')
        with (REC/(run+'.log')).open('a') as stream:
            process = subprocess.Popen([sys.executable, str(ROOT/'scripts'/script), '--run', run,
                                        '--seed', run.removeprefix('VDP-S') if run.startswith('VDP-S') else '1', '--deadline', NO_DEADLINE, *extra],
                                       cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        running.append((run, process))
    low_disk = False
    while any(process.poll() is None for _, process in running):
        write('active.json', dict(jobs={run:process.pid for run, process in running if process.poll() is None}, updated=time.time()))
        if shutil.disk_usage(WORK).free < 16*1024**3 and not low_disk:
            low_disk = True
            for _, process in running:
                if process.poll() is None:
                    try: os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError: pass
        time.sleep(5)
    assert not low_disk, 'Disk reserve reached; checkpoints and logs retained'
    assert all(process.returncode == 0 for _, process in running), str([(run,p.returncode) for run,p in running])


def done(run):
    path = OUT/'runs'/run/'execution_complete.json'
    return path.exists() and json.loads(path.read_text())['status'] == 'DONE'


def main():
    REC.mkdir(parents=True, exist_ok=True)
    lock = (REC/'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (REC/'identity.json').exists():
        raise RuntimeError('Queue already registered; inspect before restarting')
    write('identity.json', dict(pid=os.getpid(), started=time.time(), wall_clock_limit=None,
                              authorization='User: 不设定时间上限，继续服务器的任务，重新生成result',
                              training_budgets_changed=False, robot=False, final15=False))
    try:
        for run in ['NG-L-G', 'NG-F-G']:
            repair=json.loads((OUT/'runs'/run/'evaluation_repair.json').read_text())
            assert repair['status']=='PASS' and repair['step']==20000
            assert (OUT/'runs'/run/'selection.csv').exists()
        execute([(run,'p45_newgl_worker.py',['--stage','global'],False)
                 for run in ['NG-L-G','NG-F-G'] if not done(run)])
        assert all(done(run) for run in ['NG-L-G','NG-F-G'])
        execute([(run,'p45_newgl_worker.py',['--stage','local'],False)
                 for run in ['NG-L-L','NG-F-L'] if not done(run)])
        assert all(done(run) for run in ['NG-L-L','NG-F-L'])
        final.summarize()
        completed=[]
        execute([('GPU_READINESS','p45_diffusion_worker.py',['--mode','gpu-benchmark'],True)])
        assert json.loads((DIFF/'GPU_READINESS/benchmark.json').read_text())['steps']==50
        for seed in [1,2,3]:
            # All full checkpoints, export, margin and16GiB free reserve per seed.
            needed = 266622758*4*3*5 + 266622758*4 + 18*1024**3
            if shutil.disk_usage(WORK).free < needed:
                write('diffusion.json',dict(status='DEFERRED_DISK_CAPACITY',completed_seeds=completed, required_free_bytes=needed))
                break
            run=f'VDP-S{seed}'
            execute([(run,'p45_diffusion_worker.py',['--mode','train'],True)])
            state=json.loads((DIFF/run/'execution_complete.json').read_text())
            assert state['status']=='DIFFUSION_COMPLETE'
            completed.append(seed)
            write('diffusion.json',dict(status='RUNNING' if seed<3 else 'DIFFUSION_COMPLETE',completed_seeds=completed))
        write('execution_complete.json',dict(status='SERVER_TASKS_FINISHED',diffusion_completed_seeds=completed,robot='DEFERRED_BY_USER'))
    except Exception:
        (REC/'error.log').write_text(traceback.format_exc())
        write('execution_complete.json',dict(status='PARTIAL_ENGINEERING_OR_RESOURCE_STOP',robot='DEFERRED_BY_USER'))
        raise
    finally:
        final.summarize()
        with zipfile.ZipFile(WORK/'P4_5_UNBOUNDED_FINAL_result.zip','w',zipfile.ZIP_DEFLATED) as archive:
            archive.write(WORK/'P4_5_server_result.zip','P4_5_server_result.zip')
            for folder in [REC, DIFF]:
                for path in folder.rglob('*'):
                    if path.is_file() and path.suffix in ['.json','.csv','.md','.log']:
                        archive.write(path,path.relative_to(WORK.parent).as_posix())
        with zipfile.ZipFile(WORK/'P4_5_UNBOUNDED_FINAL_result.zip') as archive:
            assert archive.testzip() is None


if __name__ == '__main__':
    main()
