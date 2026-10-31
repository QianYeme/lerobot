"""SSH-independent G2 diagnostic queue; never launches policy training."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/lerobot-main')
OUT = ROOT/'experiments/p4_public_benchmark/p4_8_libero_pilot/index3/result'


def main():
    lock = (OUT/'gate_queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    stages = [('train_video_fetch', sys.executable, 'p48_r3_fetch_train_videos.py'),
              ('automatic_prototypes', '/root/autodl-tmp/p48_isolated_env/bin/python', 'p48_r3_auto_prototypes.py')]
    state = dict(controller_pid=os.getpid(), started=time.time(), status='RUNNING', stages=[])
    def save():
        path = OUT/'gate_queue_state.json';temp = path.with_suffix('.partial')
        temp.write_text(json.dumps(state, indent=2));temp.replace(path)
    save()
    for name, python, script in stages:
        record = dict(stage=name, started=time.time(), script_sha256=hashlib.sha256((ROOT/'scripts'/script).read_bytes()).hexdigest())
        state['stages'].append(record)
        with (OUT/(name+'.log')).open('a') as log:
            process = subprocess.Popen([python, '-u', str(ROOT/'scripts'/script)], env=env, cwd=ROOT,
                                       stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            record['pid'] = process.pid;save()
            record['exit_code'] = process.wait()
        record['finished'] = time.time();save()
        if record['exit_code']:
            state['status'] = 'ENGINEERING_FAILURE';save();return
    state['status'] = 'G2_DIAGNOSTICS_COMPLETE_NOT_TRAINING_AUTHORIZATION'
    state['finished'] = time.time();save()


if __name__ == '__main__':
    main()
