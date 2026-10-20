"""Detached, bounded F2.2 task supervisor. Never restarts existing training."""
import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def sessions():
    return subprocess.run(['screen','-ls'],capture_output=True,text=True).stdout


def validate_frozen(base):
    inputs = base/'inputs'
    protocol = json.loads((inputs/'protocol.json').read_text())
    assert hashlib.sha256((inputs/'gate_config.json').read_bytes()).hexdigest() == protocol['gate_sha256']
    assert hashlib.sha256((inputs/'decision.md').read_bytes()).hexdigest() == protocol['contract_sha256']
    for name, digest in json.loads((inputs/'input_hashes.json').read_text()).items():
        assert hashlib.sha256((inputs/name).read_bytes()).hexdigest() == digest, name
    for name, digest in json.loads((inputs/'source_hashes.json').read_text()).items():
        assert hashlib.sha256((base/'source'/name).read_bytes()).hexdigest() == digest, name


def run_stage(base, script):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    log = base / (script.removesuffix('.py')+'_'+stamp+'.log')
    with log.open('w') as stream:
        result = subprocess.run([sys.executable,str(base/'source/scripts'/script),'--base',str(base)],
            cwd=base/'source',stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Engineering stage failed (no scientific changes attempted): {script}; log={log}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    args = parser.parse_args()
    base=args.base.resolve()
    with (base/'autofinish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (base/'READY.json').exists():
            print('Already READY. No further task.',flush=True)
            return
        stage='training_wait'
        try:
            validate_frozen(base)
            while True:
                current={}
                active=sessions()
                for variant in ['B0','B1']:
                    directory=base/'results'/variant
                    exit_file=base/f'{variant}.exit'
                    if exit_file.exists():
                        if exit_file.read_text().strip() != '0':
                            raise RuntimeError(f'{variant} training failed; no automatic retraining/resume or extra budget permitted')
                        completion=json.loads((directory/'complete.json').read_text())
                        assert completion['steps']==100000
                        current[variant]='DONE'
                    elif f'.F22_{variant}' in active:
                        current[variant]='RUNNING'
                    elif not directory.exists() and not (base/f'{variant}.log').exists():
                        # Start only a formal variant that has NEVER started.
                        subprocess.run(['screen','-dmS',f'F22_{variant}','bash',
                            str(base/'source/scripts/train_f2_2_one.sh'),variant],check=True)
                        current[variant]='STARTING'
                    else:
                        # Existing evidence without live screen is not permission to retrain.
                        raise RuntimeError(f'{variant} has existing training evidence but no live screen/exit; preserve and investigate, never restart')
                save(base/'autofinish_state.json',dict(stage=stage,variants=current,updated_utc=datetime.now(timezone.utc).isoformat()))
                if all(s=='DONE' for s in current.values()):
                    break
                time.sleep(30)
            validate_frozen(base)
            stage='selected_dev_mechanism_and_cost'
            save(base/'autofinish_state.json',dict(stage=stage,variants=current))
            run_stage(base,'evaluate_f2_2_selected.py')
            validate_frozen(base)
            stage='audit_figures_and_package'
            save(base/'autofinish_state.json',dict(stage=stage,variants=current))
            run_stage(base,'report_f2_2.py')
            assert (base/'READY.json').exists()
            save(base/'autofinish_state.json',dict(stage='READY_STOPPED',review='WAIT_USER_AND_PLANNER',
                ready=json.loads((base/'READY.json').read_text())))
            print('READY. Stopped; no F2.3, no publication, no robot tasks.',flush=True)
        except Exception:
            detail=traceback.format_exc()
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            (base/f'autofinish_failure_{stamp}.log').write_text(detail)
            delivery=base/'delivery'; delivery.mkdir(exist_ok=True)
            save(base/'autofinish_state.json',dict(stage='BLOCKED',failed_stage=stage,error=detail,
                final_sealed=True,all_existing_evidence_preserved=True))
            (delivery/f'proposal_failure_{stamp}.md').write_text(
                '# Execution stopped\n\n'+detail+'\n\nNo scientific change applied. Preserve all evidence. '
                'Investigate engineering error without changing gate, data, metric, budget or model. '
                'Training numerical failure is not silently retried. Scientific changes require new approval.\n')
            print(detail,flush=True)
            raise


if __name__=='__main__':
    main()
