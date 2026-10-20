"""Allowlisted historical cleanup: dry-run first, apply only the saved exact inventory."""
import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path('/root/autodl-tmp/lerobot/results/experiments')
AUDIT=Path('/root/autodl-tmp/cleanup_20261019')
GROUPS=['T0_POSITION_DISCOVERY_ACT_VS_OCACT','T1_OCACT_EXPLICIT_SPATIAL_CONDITIONING',
        'T2_FIXED_SCALE_SPATIAL_CONDITIONING','T3_JOINT_STATE_SPATIAL_CONDITIONING',
        'N0_OVERNIGHT_SPATIAL_CONDITIONING_BATCH']
INVALID=ROOT/'T0_POSITION_DISCOVERY_ACT_VS_OCACT/P1_s1/checkpoints'


def inventory(path):
    if path.is_symlink():raise ValueError('Symlink target rejected: '+str(path))
    files=[path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file())
    if any(p.is_symlink() for p in path.rglob('*')) if path.is_dir() else False:
        raise ValueError('Symlink child rejected')
    return [dict(path=str(p.relative_to(path)) if p!=path else '.',size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in files]


def safe(path):
    resolved=path.resolve(strict=True)
    assert resolved.is_relative_to(ROOT.resolve()) and resolved!=ROOT.resolve()
    assert any(resolved.is_relative_to((ROOT/g).resolve()) for g in GROUPS)
    return resolved


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true')
    ap.add_argument('--audit-after-interruption',action='store_true');args=ap.parse_args()
    AUDIT.mkdir(exist_ok=True)
    plan_path=AUDIT/'plan_numeric_only.json'
    if args.audit_after_interruption:
        plan=json.loads(plan_path.read_text())
        unique={}
        for target in plan['targets']:
            path=Path(target['path'])
            resolved=path.resolve(strict=False)
            assert resolved.is_relative_to(ROOT.resolve())
            assert not path.exists(), 'Cleanup incomplete: '+str(path)
            unique.setdefault(str(resolved),dict(path=str(resolved),bytes=sum(f['size'] for f in target['files'])))
        report=dict(timestamp=datetime.now(timezone.utc).isoformat(),status='DELETED_TARGETS_VERIFIED_AFTER_ALIAS_ERROR',
            deleted=list(unique.values()),logical_bytes=sum(t['bytes'] for t in unique.values()),
            free_after=shutil.disk_usage(ROOT).free,free_before=None,
            note='All unique targets removed. Final invalid-model last alias duplicated its numeric checkpoint and raised FileNotFoundError after deletion; audit verifies absence without further deletion. Initial plan rejected nonnumeric training-state aliases before deleting anything.',
            recovery='No trash backup; intermediate optimizer states need external backup or retraining to restore.',protected=plan['protected'])
        (AUDIT/'deletion_report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(dict(unique_targets=len(unique),GiB=report['logical_bytes']/2**30,free_after_GiB=report['free_after']/2**30)))
        return
    if not args.apply:
        targets=[]
        for group in GROUPS:
            for path in sorted((ROOT/group).glob('*/checkpoints/*/training_state')):
                if not path.parent.name.isdigit() or path.parent.name=='100000' or path.parent.is_symlink():
                    continue # retain final state and every `last` alias
                safe(path)
                targets.append(dict(path=str(path),kind='historical_intermediate_optimizer_state',files=inventory(path)))
        if INVALID.exists():
            for path in sorted(INVALID.glob('*/pretrained_model/model.safetensors')):
                if not path.parent.parent.name.isdigit() or path.parent.parent.is_symlink():
                    continue
                safe(path)
                targets.append(dict(path=str(path),kind='INVALID_P1_s1_omitted_object_module',files=inventory(path)))
        plan=dict(created=datetime.now(timezone.utc).isoformat(),targets=targets,
            bytes=sum(f['size'] for t in targets for f in t['files']),
            protected='all valid model weights, 100k training states, all F2.1 files, datasets, code, logs, P1_ocfix2, reports')
        if plan_path.exists():raise RuntimeError('Do not overwrite previous deletion plan')
        plan_path.write_text(json.dumps(plan,indent=2))
        print(json.dumps(dict(targets=len(targets),GiB=plan['bytes']/2**30,plan=str(plan_path))))
        return
    plan=json.loads(plan_path.read_text())
    assert not (AUDIT/'deletion_report.json').exists()
    # Validate every exact resolved target and unchanged inventory before deleting any.
    for target in plan['targets']:
        path=Path(target['path']);safe(path)
        assert inventory(path)==target['files'],'Changed since dry-run: '+str(path)
        if target['kind']=='historical_intermediate_optimizer_state':
            assert path.name=='training_state' and path.parent.name.isdigit() and path.parent.name!='100000'
        else:
            assert path.is_relative_to(INVALID) and path.name=='model.safetensors'
    before=shutil.disk_usage(ROOT).free
    deleted=[]
    for target in plan['targets']:
        path=Path(target['path'])
        if path.is_file():path.unlink()
        else:shutil.rmtree(path)
        deleted.append(dict(path=str(path),bytes=sum(f['size'] for f in target['files'])))
    os.sync()
    report=dict(timestamp=datetime.now(timezone.utc).isoformat(),deleted=deleted,
        logical_bytes=sum(t['bytes'] for t in deleted),free_before=before,free_after=shutil.disk_usage(ROOT).free,
        recovery='No trash backup: intermediate optimizer states not recoverable without external backup/retraining. Invalid model excluded from science; metadata/logs retained.',
        protected=plan['protected'])
    (AUDIT/'deletion_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(targets=len(deleted),logical_GiB=report['logical_bytes']/2**30,
        free_before_GiB=before/2**30,free_after_GiB=report['free_after']/2**30)))


if __name__=='__main__':main()
