"""Explicit, credential-scanned staging of force research code/reports, never weights."""
import json
import os
import re
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
prefixes=['scripts/','tests/scripts/','experiments/force/f1rm_teleop_load_retrospective_match/',
    'experiments/force/f2_1_load_temporal_encoding_ablation/','experiments/force/f2_2_contact_gated_single_frame_load/',
    'experiments/force/f2_3_global_vs_local_load_regulation/','experiments/force/f3_real_robot_local_regulator_feasibility/']
allowed={'.py','.sh','.md','.json','.csv','.png','.txt','.html','.toml'}
raw=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=ROOT).decode('utf-8').split('\0')
raw+=subprocess.check_output(['git','diff','--name-only','-z'],cwd=ROOT).decode('utf-8').split('\0')
paths=['workflow/STATUS.md','docc/日志/工作日志.md','src/lerobot/policies/act/configuration_act.py',
       'src/lerobot/policies/act/modeling_act.py','src/lerobot/policies/act/local_load_regulator.py']
for name in raw:
    if not name or not any(name.startswith(prefix) for prefix in prefixes):continue
    if Path(name).suffix.lower() not in allowed:continue
    if any(piece.startswith('pretraining') for piece in Path(name).parts):continue
    if '/csv_backups/' in name or '/__pycache__/' in name:continue
    paths.append(name)
paths=sorted(set(paths))
for name in paths:
    data=(ROOT/name).read_bytes()
    assert len(data)<10*1024*1024, 'Large artifact requires review: '+name
    assert not re.search(rb'hf_[A-Za-z0-9]{20,}',data), 'Possible HF credential: '+name
    for key in ['D2_PASSWORD','HF_TOKEN']:
        value=os.environ.get(key,'').encode()
        assert not value or value not in data, 'Credential found: '+name
for offset in range(0,len(paths),80):
    subprocess.run(['git','add','--',*paths[offset:offset+80]],cwd=ROOT,check=True)
(ROOT/'outputs/force_release_staging.json').write_text(json.dumps(dict(paths=paths,count=len(paths)),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(staged_files=len(paths),payload_bytes=sum((ROOT/name).stat().st_size for name in paths))))
