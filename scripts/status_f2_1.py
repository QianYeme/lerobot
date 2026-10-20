"""Read-only process/training audit; store an evidence snapshot, never stop jobs."""
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

BASE=Path('/root/autodl-tmp/F2_1_r1_20261019')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    variants={}
    active_path=BASE/'active_runs.json'
    active=json.loads(active_path.read_text()) if active_path.exists() else {}
    for name in ['A0','A1','A2']:
        run=active.get(name,{})
        output=Path(run.get('output_dir',str(BASE/'results'/name)))
        log=output/'training.csv'
        rows=list(csv.DictReader(log.open())) if log.exists() else []
        exit_path=BASE/run.get('exit_file',name+'.exit')
        variants[name]=dict(logged_steps=int(rows[-1]['step']) if rows else 0,
            last_log=rows[-1] if rows else None,exit_code=exit_path.read_text().strip() if exit_path.exists() else None,
            complete=(output/'complete.json').exists(),screen=run.get('screen','F21_'+name),
            output_dir=str(output),attempt=run.get('attempt',1),previous_attempt=run.get('previous_attempt'))
    keys=['source/src/lerobot/policies/act/modeling_act.py','source/src/lerobot/policies/act/configuration_act.py',
          'source/scripts/run_f2_1.py','inputs/normalization.json','inputs/split_manifest.json','inputs/cup_prototype_bank_v1.pt']
    snapshot=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),base=str(BASE),variants=variants,
        disk_free_bytes=shutil.disk_usage(BASE).free,source_sha256={name:sha(BASE/name) for name in keys},
        dataset_parquet_sha256=sha(Path('/root/autodl-tmp/lerobot/formal4_data/kind_merged_nomaster/data/chunk-000/file-000.parquet')),
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),
        screens=subprocess.run(['screen','-ls'],capture_output=True,text=True).stdout,
        torch_version=subprocess.check_output(['/root/miniconda3/bin/python','-m','pip','show','torch'],text=True).split('Version: ')[1].splitlines()[0],
        baseline_forward=json.loads((BASE/'results/historical_metric_reproduction.json').read_text()),
        smoke=json.loads((BASE/'results/smoke.json').read_text()))
    (BASE/'results/runtime_snapshot.json').write_text(json.dumps(snapshot,indent=2))
    print(json.dumps(snapshot,indent=2))


if __name__=='__main__':main()
