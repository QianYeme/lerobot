"""Exercise all four existing routes with synthetic input, never hardware."""
import csv
import json
import os
import subprocess
from pathlib import Path

W=Path('/root/autodl-tmp/R5_11_20261025')


def main():
    env=dict(os.environ,PYTHONPATH=str(W/'source/src'))
    roots={'R0':'/root/autodl-tmp/F5_20261022/bundle','R1':'/root/autodl-tmp/F5_20261022/bundle',
        'R2':'/root/autodl-tmp/R5_10_20261025/bundle','R3':str(W/'bundle')}
    rows=[]
    for model,bundle in roots.items():
        with (W/f'route_{model}.log').open('w') as f:
            subprocess.run(['/root/miniconda3/bin/python',str(W/'source/src/lerobot/scripts/control_r5_11.py'),
                '--bundle',bundle,'--session-dir',str(W/'route_smoke'),'--dry-run','--model',model,'--position','A','--policy.device=cuda'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        output=json.loads((W/f'route_smoke/offline_{model}.json').read_text())
        assert output['hardware']=='NOT_CONNECTED' and len(output['action'])==6
        identity=output['identity']
        rows.append(dict(model=model,global_sha256=identity['global_sha256'],local_sha256=identity.get('local_sha256'),
            state_dimension=len(identity['state_fields']),global_load=identity['global_load'],bundle=bundle,
            normalization_path=identity.get('normalization_path','normalization.json'),scope='SYNTHETIC_CODE_SMOKE_ONLY'))
        print(model,'PASS_SYNTHETIC_NO_HARDWARE',flush=True)
    with (W/'model_identity_manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (W/'route_smoke.json').write_text(json.dumps(dict(status='PASS_SYNTHETIC_ONLY',routes=rows,robot_trials=0),indent=2)+'\n')


if __name__=='__main__':main()
