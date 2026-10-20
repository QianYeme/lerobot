"""Wait for the bounded diagnostic capture and repeat it; never alter formal training."""
import json
import os
import subprocess
import time
from pathlib import Path

BASE=Path('/root/autodl-tmp/F2_1_r1_20261019')
OUT=BASE/'diagnosis_A0'
env=dict(os.environ,PYTHONPATH=str(BASE/'source/src'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
started=time.monotonic()
report=None
while time.monotonic()-started<1900:
    if (OUT/'failure_report.json').exists():
        capture=json.loads((OUT/'failure_report.json').read_text())
        repetitions=[]
        for n in range(1,4):
            folder=BASE/f'diagnosis_A0_exact_{n}'
            command=['/root/miniconda3/bin/python',str(BASE/'source/scripts/diagnose_f2_1_a0_replay.py'),
                '--root',str(BASE/'inputs'),'--data','/root/autodl-tmp/lerobot/formal4_data/kind_merged_nomaster',
                '--out',str(folder),'--replay-capture',str(OUT/'failure_capture.pt')]
            with (BASE/f'diagnosis_A0_exact_{n}.log').open('w') as stream:
                result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=env)
            path=folder/'capture_replay.json'
            repetitions.append(json.loads(path.read_text()) if path.exists() else dict(process_exit=result.returncode))
        grads=capture['gradients']
        if grads['all_gradient_elements_finite'] and grads.get('norm_fp32')=='NONFINITE':
            category='CAPTURED_FINITE_GRADIENT_FP32_NORM_OVERFLOW'
        else:
            category='CAPTURED_NONFINITE_PARAMETER_GRADIENTS'
        report=dict(status=category,captured_step=capture['step'],replays=repetitions,
                    original_run_exact_identity='UNAVAILABLE_ORIGINAL_FAILED_STATE_NOT_SAVED',
                    formal_training_modified=False,fix_applied=False)
        break
    if (OUT/'no_reproduction.json').exists():
        report=json.loads((OUT/'no_reproduction.json').read_text())
        break
    exit_path=BASE/'diagnosis_A0.exit'
    if exit_path.exists() and exit_path.read_text().strip()!='0':
        report=dict(status='DIAGNOSTIC_PROCESS_FAILED',exit_code=exit_path.read_text().strip(),fix_applied=False)
        break
    time.sleep(10)
if report is None:report=dict(status='WATCH_BUDGET_EXPIRED_NO_CONCLUSION',fix_applied=False)
(BASE/'diagnosis_A0_summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
