"""Server-only bounded F3 artifact export/upload. Credentials only from HF_TOKEN."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

BASE=Path('/root/autodl-tmp/F2_3_r0_20261020')
TRANSFER=Path('/root/autodl-tmp/F3_transfer_20261020')
BUNDLE=TRANSFER/'bundle'
REPO='QYyyyyyyy/F3_C1_LOAD_SO101'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False))


def prepare():
    BUNDLE.mkdir(parents=True,exist_ok=False)
    identity=json.loads((BASE/'result/global_checkpoint_identity.json').read_text())
    checkpoint=Path(identity['checkpoint'])
    assert sha(checkpoint/'model.safetensors')==identity['model_sha256']
    assert identity['step']==80000
    local=BASE/'result/C1-load/010000.pt'
    saved=torch.load(local,map_location='cpu',weights_only=True)
    assert saved['variant']=='C1-load' and saved['step']==10000
    assert saved['config']['delta_max']==4.832699459791184
    shutil.copytree(checkpoint,BUNDLE/'global')
    (BUNDLE/'local').mkdir();shutil.copy2(local,BUNDLE/'local/010000.pt')
    for source,name in [(BASE/'inputs/normalization.json','normalization.json'),
        (BASE/'result/local_normalization.json','local_normalization.json'),
        (BASE/'result/residual_config.json','residual_config.json'),
        (BASE/'result/global_checkpoint_identity.json','global_checkpoint_identity.json')]:
        shutil.copy2(source,BUNDLE/name)
    (BUNDLE/'assets').mkdir();shutil.copy2(BASE/'inputs/cup_prototype_bank_v1.pt',BUNDLE/'assets/cup_prototype_bank_v1.pt')
    shutil.copytree(BASE/'source/src',BUNDLE/'source/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(TRANSFER/'local_load_regulator.py',BUNDLE/'source/src/lerobot/policies/act/local_load_regulator.py')
    (BUNDLE/'source/scripts').mkdir();shutil.copy2(TRANSFER/'verify_f3_bundle.py',BUNDLE/'source/scripts/verify_f3_bundle.py')
    for name in ['pyproject.toml','TRANSFER_GUIDE.md','decision.md']:
        shutil.copy2(TRANSFER/name,BUNDLE/('source/pyproject.toml' if name=='pyproject.toml' else name))
    (BUNDLE/'torch_cache/hub/checkpoints').mkdir(parents=True)
    hub=Path('/root/.cache/torch/hub')
    for name in ['dinov2_vits14_pretrain.pth','resnet18-f37072fd.pth']:
        shutil.copy2(hub/'checkpoints'/name,BUNDLE/'torch_cache/hub/checkpoints'/name)
    shutil.copytree(hub/'facebookresearch_dinov2_main',BUNDLE/'torch_cache/hub/facebookresearch_dinov2_main',
        ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
    with np.load(BASE/'cache/dev.npz') as z, np.load(BASE/'result/C1-load/selected_dev.npz') as prediction:
        norm=json.loads((BASE/'inputs/normalization.json').read_text())
        nominal=z['nominal'][:16]*np.asarray(norm['action']['std'])+np.asarray(norm['action']['mean'])
        (BUNDLE/'validation').mkdir()
        np.savez_compressed(BUNDLE/'validation/local_fixture.npz',nominal_native=nominal.astype(np.float32),
            raw_load=z['load'][:16]*norm['load']['std']+norm['load']['mean'],position=z['position'][:16],
            expected_delta=prediction['delta'][:16])
    versions=dict(python=platform.python_version(),platform=platform.platform())
    for name in ['torch','torchvision','numpy','huggingface-hub','safetensors','transformers']:
        try:
            versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name]='not installed in validation runtime'
    save(BUNDLE/'environment_versions.json',versions)
    save(BUNDLE/'regulator_config.json',dict(default_mode='disabled',bound=saved['config']['delta_max'],
        hardware_integration='PENDING',deployment_hardware_stage0='NOT_RUN',robot_trials='NOT_RUN',F3_COMPLETE=False))
    save(BUNDLE/'local_checkpoint_identity.json',dict(variant='C1-load',step=10000,sha256=sha(local),
        source=str(local),global_sha256=identity['model_sha256'],architecture='3-32-32-1 GELU/tanh',seed=1))
    (BUNDLE/'README.md').write_text('---\nlicense: apache-2.0\ntags:\n- robotics\n- lerobot\n- research\n---\n\n'
        '# F3 C1-load SO101 migration bundle\n\nExact frozen F2.3 A0-80k global + C1-load-10k local. '
        'Offline exploratory artifacts; NOT hardware validated. Default disabled. '
        'See TRANSFER_GUIDE.md and decision.md before use. F3 robot trials NOT RUN. '
        'No complete robot controller integration yet. Original third-party cache licenses remain in torch_cache.\n')
    save(BUNDLE/'manifest.json',dict(repo_id=REPO,global_sha256=identity['model_sha256'],local_sha256=sha(local),
        sha256={str(path.relative_to(BUNDLE)):sha(path) for path in sorted(BUNDLE.rglob('*')) if path.is_file()}))
    env=dict(os.environ,PYTHONPATH=str(BUNDLE/'source/src'),TORCH_HOME=str(BUNDLE/'torch_cache'))
    with (TRANSFER/'offline_validation.log').open('w') as stream:
        subprocess.run(['/root/miniconda3/bin/python',str(BUNDLE/'source/scripts/verify_f3_bundle.py'),
            '--bundle',str(BUNDLE),'--device','cuda'],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    shutil.copy2(TRANSFER/'offline_validation.log',BUNDLE/'validation/offline_validation.log')
    # Manifest includes all released payloads except itself.
    manifest=json.loads((BUNDLE/'manifest.json').read_text())
    manifest['sha256']['validation/offline_validation.log']=sha(BUNDLE/'validation/offline_validation.log')
    save(BUNDLE/'manifest.json',manifest)
    print('PREPARED_OFFLINE_VALIDATED',REPO,flush=True)


def upload():
    from huggingface_hub import HfApi, hf_hub_download
    api=HfApi(endpoint='https://huggingface.co',token=os.environ['HF_TOKEN'])
    print('AUTH_BEGIN',flush=True)
    account=api.whoami()['name'];assert account=='QYyyyyyyy',account
    print('AUTH_OK_CREATE_REPO',flush=True)
    api.create_repo(REPO,repo_type='model',private=True,exist_ok=True)
    print('UPLOAD_BEGIN',flush=True)
    commit=api.upload_folder(repo_id=REPO,repo_type='model',folder_path=str(BUNDLE),
        commit_message='Publish exact F3 C1-load migration bundle; hardware validation pending')
    revision=commit.oid
    manifest=json.loads((BUNDLE/'manifest.json').read_text())
    remote=Path(hf_hub_download(REPO,'manifest.json',revision=revision,token=os.environ['HF_TOKEN']))
    assert remote.read_bytes()==(BUNDLE/'manifest.json').read_bytes()
    info=api.model_info(REPO,revision=revision,files_metadata=True)
    files={f.rfilename:f for f in info.siblings}
    for name in ['global/model.safetensors','local/010000.pt','assets/cup_prototype_bank_v1.pt']:
        record=files[name];lfs=record.lfs
        if lfs is not None:
            assert lfs.sha256==manifest['sha256'][name],name
        else:
            downloaded=Path(hf_hub_download(REPO,name,revision=revision,token=os.environ['HF_TOKEN']))
            assert sha(downloaded)==manifest['sha256'][name],name
    save(TRANSFER/'publication.json',dict(repo_id=REPO,revision=revision,url='https://huggingface.co/'+REPO,
        private=info.private,manifest_sha256=sha(BUNDLE/'manifest.json'),critical_model_hashes_verified=True,
        hardware_validation='NOT_RUN',robot_trials='NOT_RUN'))
    print((TRANSFER/'publication.json').read_text(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['prepare','upload'],required=True)
    args=parser.parse_args();prepare() if args.mode=='prepare' else upload()
