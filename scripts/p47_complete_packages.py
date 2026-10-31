"""Complete self-contained inference packages, without publishing or credentials."""
import json
from pathlib import Path
import shutil
import subprocess

import torch

from p47_stage_models import ROOT,P4,WORK,OLD,SERVER,DIFF,sha


def main():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    inventory=json.loads((WORK/'result/upload/model_inventory.json').read_text())
    for record in inventory:
        role=record['role'];dest=WORK/'staging'/role
        stock=role in ['VACT-S1','VDP-S1']
        source=P4/'p4_5_robot_reliability_server10h/index1'/('stock' if stock else 'newgl_code')
        shutil.copytree(source/'src',dest/'source/src',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        # Add only deployment support modules; policy/processor source remains frozen.
        for rel in ['lerobot/scripts/control_f5.py','lerobot/utils/f5_protocol.py']:
            target=dest/'source/src'/rel;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/'src'/rel,target)
        for name in ['p47_runtime.py','control_p47.py']:
            shutil.copy2(ROOT/'scripts'/name,dest/name)
        shutil.copy2(ROOT/'LICENSE',dest/'LICENSE')
        assets=dest/'assets';assets.mkdir(exist_ok=True)
        original=P4/'p4_1_unattended_p1_pilot/index3/result'
        stats=torch.load(original/'train90_statistics.pt',weights_only=True)
        if role=='VDP-S1':stats=torch.load(DIFF/'VDP-S1/statistics.pt',weights_only=True)
        elif role not in ['L1','F3','A1']:
            full=stats
            stats={k:({name:value[:6].clone() for name,value in v.items()} if k=='observation.state' else v) for k,v in full.items()}
            if role.startswith('NG-'):
                torch.save(dict(mean=full['observation.state']['mean'][6],std=full['observation.state']['std'][6]),assets/'load_statistics.pt')
        torch.save(stats,assets/'statistics.pt')
        if role not in ['VACT-S1','VDP-S1','A1']:
            shutil.copy2(original/'formal6_train90_prototypes.pt',assets/'formal6_train90_prototypes.pt')
        cache=dest/'torch_cache/hub';cache.mkdir(parents=True,exist_ok=True)
        needed=['resnet18-f37072fd.pth']
        if role not in ['VACT-S1','VDP-S1','A1']:
            shutil.copytree(OLD/'torch_cache/hub/facebookresearch_dinov2_main',cache/'facebookresearch_dinov2_main',dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('__pycache__','.git','*.pyc'))
            needed.append('dinov2_vits14_pretrain.pth')
        (cache/'checkpoints').mkdir(exist_ok=True)
        for name in needed:shutil.copy2(OLD/'torch_cache/hub/checkpoints'/name,cache/'checkpoints'/name)
        record.update(git_commit=commit,official_stock=stock,official_commit='8fff0fde7c79f23a93d845d1a50e985de01f8b8a' if stock else None,
                      dataset_revision='83a829af6df245ed53596dbc22b18c19f8bd3315',split='Train90; Dev15; no Final15',
                      state_fields=['shoulder_pan.pos','shoulder_lift.pos','elbow_flex.pos','wrist_flex.pos','wrist_roll.pos','gripper.pos']+(['gripper.load','gripper.curr'] if role in ['L1','F3','A1'] else []),
                      separate_load=role.startswith('NG-'),double_norm=role in ['L1','NG-L-GL'],
                      units='arm degrees; gripper normalized0..100; same nn calibration',
                      local_application='Once after temporal ensemble and global denormalization; before relative limit',
                      upload_status='NOT_UPLOADED',inference_validation='PENDING',
                      authentication_override='User explicitly authorized supplied credential; token value never packaged')
        (dest/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
        (dest/'README.md').write_text(f'''---
license: apache-2.0
tags: [lerobot, so101, robotics]
---
# {role}

Private deployment package; not robot validated. Source checkpoint and hashes: manifest.json.
Train90 formal6 fixed revision; Dev15 development evidence only. No Final15 used.
{'Official v0.4.4 policy/processor source, unchanged.' if stock else 'Project ACT/OC-ACT source snapshot; see preprocessing contract in manifest.'}
{record['step']} global updates; {record.get('local_step',0)} Local updates. Local OFF/ON shares the exact Global.
No optimizer, RNG state, training videos or robot calibration are included.

Runtime: Python3.12.3, torch2.8.0+cu128, torchvision0.23.0, torchcodec0.7.0.
Use the bundled source/src via PYTHONPATH, not another installed lerobot policy.
Run `python control_p47.py --help --bundle .` for the supervised single-trial entrypoint.
Hardware: SO101 follower nn, ACM0, top4/gripper2, RGB640x480@30; arm degrees, gripper0..100.
Keep the same nn calibration and manual Home+Open. Never run unattended robot motions.
ACT deployment settings1 action/.01 ensemble are explicit CLI overrides, not training defaults.
Diffusion retains native2 observations/16 prediction/8 executed actions,100DDPM; no ACT ensemble.
New-GL defaults Local OFF; --local-on enables the bounded gripper-only correction.
Absent cup must use --shadow-no-cup; this branch records predictions and never calls send_action.
See REAL_ROBOT_TEST_GUIDE.md for download instructions and full commands once upload verification completes.
''',encoding='utf-8')
        checks=[dict(path=p.relative_to(dest).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in dest.rglob('*') if p.is_file() and p.name!='checksums.json']
        (dest/'checksums.json').write_text(json.dumps(checks,indent=2))
        print(role,'PACKAGE_STAGED',flush=True)


if __name__=='__main__':main()
