"""Publish verified inference assets. Supply HF_TOKEN only through the environment."""
import hashlib
import json
import os
import zipfile

from huggingface_hub import HfApi
from p43_export_robot_bundle import ROOT, OUT


def main():
    preflight=json.loads((OUT/'deployment_preflight.json').read_text())
    assert preflight['status']=='OFFLINE_PASS_HARDWARE_PENDING'
    assert preflight['models']==['L1','F3','A1','L1_160K']
    with zipfile.ZipFile(OUT/'runtime_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in (ROOT/'src').rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
                archive.write(path,str(path.relative_to(ROOT)))
        for name in ['pyproject.toml','README.md','LICENSE']:
            archive.write(ROOT/name,name)
    (OUT/'README.md').write_text('''---
license: apache-2.0
tags:
- lerobot
- act
- so101
---
# P4.3 formal6 SO101: four real-robot candidates

L1: Legacy P1 seed1 80k. F3: Fixed raw-Top P1 seed3 80k.
A1: Original ACT seed1 80k. L1_160K: continued Legacy P1 seed1 160k.

See [Chinese test guide](REAL_ROBOT_TEST_GUIDE.md) for four separate commands.
Includes losslessly exported inference weights, Train90 statistics and prototypes,
DINO/ResNet offline cache, and matching runtime source snapshot.
Training optimizer states remain on the training server and are not needed here.
No dataset video or robot calibration is included; retain your own matching nn calibration.

Offline wrapper checks are not real-robot validation. Dev15 uses teacher-forced observations.
The original 36-trial block is retained, with a separate user-authorized 12-trial 160k supplement.
No real robot has been operated by this preparation task.
''',encoding='utf-8')
    manifest=[]
    for path in sorted(OUT.rglob('*')):
        if (path.is_file() and '.cache' not in path.parts and '__pycache__' not in path.parts
                and path.suffix!='.pyc' and path.name!='delivery_manifest.json'):
            with path.open('rb') as stream:
                digest=hashlib.file_digest(stream,'sha256').hexdigest()
            manifest.append(dict(path=path.relative_to(OUT).as_posix(),bytes=path.stat().st_size,sha256=digest))
    (OUT/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    api=HfApi(token=os.environ['HF_TOKEN'])
    repo='QYyyyyyyy/P4_3_FORMAL6_SO101'
    api.create_repo(repo_id=repo,repo_type='model',exist_ok=True,private=False)
    commit=api.upload_folder(repo_id=repo,folder_path=str(OUT),repo_type='model',
        ignore_patterns=['.cache/**','**/__pycache__/**','*.pyc'],
        commit_message='Publish L1 80k, F3 80k, ACT 80k and continued P1 160k robot bundle')
    info=api.model_info(repo,revision=commit.oid,files_metadata=True)
    remote={item.rfilename:item for item in info.siblings}
    for item in manifest:
        entry=remote[item['path']]
        assert entry.size==item['bytes'],item['path']
        if entry.lfs:
            assert entry.lfs.sha256==item['sha256'],item['path']
    receipt=dict(repo=repo,url=f'https://huggingface.co/{repo}',commit=commit.oid,
        files=len(manifest)+1,remote_sizes_verified=True,remote_lfs_hashes_verified=True)
    (OUT.parent/'huggingface_upload_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
