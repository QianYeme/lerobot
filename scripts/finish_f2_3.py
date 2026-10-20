"""Detached F2.3 cache wait -> matched local training -> report; bounded scope."""
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

BASE=Path('/root/autodl-tmp/F2_3_r0_20261020')


def save(path,value):path.write_text(json.dumps(value,indent=2))


def main():
    source=BASE/'source';out=BASE/'result'
    if (BASE/'READY.json').exists():return
    snapshot={str(path.relative_to(BASE)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [
        BASE/'inputs/residual_config.json',BASE/'inputs/baseline_identity.json',BASE/'inputs/decision.md',
        BASE/'inputs/normalization.json',BASE/'inputs/split_manifest.json',source/'scripts/run_f2_3.py',
        source/'scripts/report_f2_3.py',source/'src/lerobot/policies/act/configuration_act.py',source/'src/lerobot/policies/act/modeling_act.py']}
    save(out/'runtime_hashes.json',snapshot)
    save(out/'reporting_interpretation.json',dict(meaningful_episode_consistency_min=8,
        predominantly_nonsaturated_near_bound_max=.2,saturation_dominates_fraction=.5,
        diagnostic_nonzero_floor=1e-6,acceptable_latency_ratio_max=1.2,
        note='Conservative descriptive categories frozen before local training; no change to metric or checkpoint selection'))
    stage='replay'
    try:
        while not (out/'replay_complete.json').exists():
            sessions=subprocess.run(['screen','-ls'],capture_output=True,text=True).stdout
            if '.F23_CACHE' not in sessions:
                raise RuntimeError('Replay stopped without PASS; inspect cache.log and cache_start_screen.txt')
            save(BASE/'state.json',dict(stage=stage,status='RUNNING'));time.sleep(15)
        for name,expected in snapshot.items():assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==expected,name
        stage='local_training'
        save(BASE/'state.json',dict(stage=stage,status='RUNNING'))
        for name,args in [('training',['run_f2_3.py','--stage','train']),('report',['report_f2_3.py'])]:
            stage=name
            with (BASE/f'{name}.log').open('x') as stream:
                subprocess.run([sys.executable,str(source/'scripts'/args[0]),'--base',str(BASE)]+args[1:],
                    cwd=source,stdout=stream,stderr=subprocess.STDOUT,check=True)
            for path,expected in snapshot.items():assert hashlib.sha256((BASE/path).read_bytes()).hexdigest()==expected,path
        assert (BASE/'READY.json').exists()
        save(BASE/'state.json',dict(status='READY_STOPPED',ready=json.loads((BASE/'READY.json').read_text())))
    except Exception:
        error=traceback.format_exc();print(error,flush=True)
        (BASE/'failure.log').write_text(error)
        save(BASE/'state.json',dict(status='BLOCKED',stage=stage,error=error))
        (out/'proposal.md').write_text('# Execution stopped\n\n'+error+'\n\nPreserve all evidence. No automatic retraining or scientific changes.\n')
        raise


if __name__=='__main__':main()
