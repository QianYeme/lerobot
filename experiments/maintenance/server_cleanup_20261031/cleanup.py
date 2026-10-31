"""Authorized cleanup: historical intermediate resume files and rebuildable caches."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import shutil
import subprocess
import datetime

ROOT = Path('/root/autodl-tmp/lerobot-main')
OUT = ROOT / 'experiments/maintenance/server_cleanup_20261031'
OUT.mkdir(parents=True, exist_ok=True)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')

processes = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
active = [line for line in processes.splitlines()
          if 'python' in line and any(name in line for name in
             ['p48_r6_queue.py', 'p48_r5_queue.py', 'lerobot_train', 'p45_train.py'])]
assert not active, active
before = {str(p): shutil.disk_usage(p)._asdict() for p in [ROOT, Path('/')]}
groups = defaultdict(list)
historical = ROOT / 'experiments/p4_final_p1_training'
for p in historical.rglob('resume.pt'):
    if p.parent.name.isdigit():
        groups[p.parent.parent].append(p)
delete = []
preserve = []
for group, paths in sorted(groups.items()):
    newest = max(paths, key=lambda p: int(p.parent.name))
    for p in sorted(paths):
        # Keep final recovery state, every 80k candidate, and standalone smoke evidence.
        if p == newest or p.parent.name == '080000':
            preserve.append(p)
        else:
            delete.append((p, 'historical intermediate resume; latest/80k retained'))

# Preserve all runtime environments, DINO caches, datasets, formal r5 checkpoints,
# r5 smoke/resume evidence, videos, source snapshots, and published robot exports.
cache_roots = [Path('/root/.cache/pip'), Path('/root/.cache/huggingface/xet'),
               ROOT / 'outputs/so101_purchase_20260930/node_modules']
for directory in cache_roots:
    if directory.exists():
        delete.extend((p, 'rebuildable package/download cache') for p in directory.rglob('*')
                      if p.is_file() and not p.is_symlink())

allowed = [historical.resolve()] + [p.resolve() for p in cache_roots]
records = []
for p, reason in delete:
    assert not p.is_symlink() and p.is_file()
    resolved = p.resolve()
    assert any(resolved.is_relative_to(a) for a in allowed), str(p)
    assert 'p4_public_benchmark' not in str(p)
    records.append({'path': str(p), 'bytes': p.stat().st_size,
                    'sha256': digest(p), 'reason': reason})
kept = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in preserve]
write('deletion_plan.json', {'before': before, 'files': records, 'retained_resume_files': kept})
for item in records:
    p = Path(item['path'])
    assert p.stat().st_size == item['bytes']
    p.unlink()
for item in kept:
    assert digest(Path(item['path'])) == item['sha256']

r5 = ROOT / 'experiments/p4_public_benchmark/p4_8_libero_pilot/index5'
manifest = json.loads((r5 / 'result/artifact_manifest.json').read_text())
for item in manifest.values():
    assert digest(Path(item['server_path'])) == item['sha256'], item['server_path']
assert all(not Path(item['path']).exists() for item in records)
after = {str(p): shutil.disk_usage(p)._asdict() for p in [ROOT, Path('/')]}
report = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'status': 'DONE', 'deleted_files': len(records),
          'deleted_logical_bytes': sum(item['bytes'] for item in records),
          'deleted_resume_files': sum(item['path'].endswith('/resume.pt') for item in records),
          'retained_resume_files': len(kept), 'retained_resume_sha': 'PASS',
          'r5_manifest_assets_verified_unchanged': len(manifest),
          'before': before, 'after': after,
          'data_disk_freed_bytes': after[str(ROOT)]['free'] - before[str(ROOT)]['free'],
          'raw_logs_videos_datasets_robot_exports': 'PRESERVED'}
write('cleanup_report.json', report)
print(json.dumps(report, indent=2), flush=True)
