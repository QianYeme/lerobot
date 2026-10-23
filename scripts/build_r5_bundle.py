"""Build exact R5 model payload on the training server; never execute hardware."""
import json
import os
import shutil
import tarfile
from pathlib import Path

from lerobot.utils.r5_protocol import MODELS,LABELS,EXPECTED,POSITION_FIELDS,BOUND,verify_bundle,digest

WORK=Path('/root/autodl-tmp/R5_20261023')
F5=Path('/root/autodl-tmp/F5_1_r3_20261022')
OLD=Path('/root/autodl-tmp/F4_1_20261022/runs/G-Load/seed3/attempt1/checkpoints/080000/pretrained_model')


def main():
    if shutil.disk_usage('/root/autodl-tmp').free<18*1024**3:raise RuntimeError('Preserve15GiB reserve')
    root=WORK/'bundle';root.mkdir(exist_ok=False)
    paths={'G36-S3':OLD,'G60-S3':F5/'runs/G60-S3/checkpoints/080000/pretrained_model',
        'G60-S4':F5/'runs/G60-S4/checkpoints/080000/pretrained_model'}
    for name,path in paths.items():
        destination=root/'models'/name;destination.mkdir(parents=True)
        for filename in ['config.json','model.safetensors','f2_1_normalization.json']:
            os.link(path/filename,destination/filename)
    (root/'normalization').mkdir()
    shutil.copy2(OLD/'f2_1_normalization.json',root/'normalization/train36.json')
    shutil.copy2(F5/'inputs/normalization.json',root/'normalization/full60.json')
    (root/'local').mkdir()
    local_paths={'New-GL-36':Path('/root/autodl-tmp/F4_2_20261022/result/New-GL/seed3/007000.pt'),
        'New-GL-60':F5/'runs/L60-S3/attempt1/007000.pt'}
    for name,path in local_paths.items():os.link(path,root/'local'/f'{name}.pt')
    source=Path('/root/autodl-tmp/F5_20261022/bundle')
    shutil.copytree(source/'assets',root/'assets',copy_function=os.link)
    shutil.copytree(source/'torch_cache',root/'torch_cache',copy_function=os.link,ignore=shutil.ignore_patterns('__pycache__','.git'))
    identities={}
    for model in MODELS:
        global_name='G36-S3' if model in ['M1','M3'] else 'G60-S4' if model=='M5' else 'G60-S3'
        global_sha,local_sha=EXPECTED[model]
        identities[model]=dict(model_label=LABELS[model],global_path='models/'+global_name,global_sha256=global_sha,
            normalization_path='normalization/'+('train36.json' if model in ['M1','M3'] else 'full60.json'),
            state_fields=POSITION_FIELDS,global_load=True,global_step=80000,global_seed=4 if model=='M5' else 3)
        if local_sha:identities[model].update(local_path=f'local/{LABELS[model]}.pt',local_sha256=local_sha,bound=BOUND,local_step=7000,local_seed=3)
    (root/'model_identity.json').write_text(json.dumps(identities,indent=2)+'\n')
    (root/'README.md').write_text('R5 five frozen routes. Use control_r5.py; cudnn.benchmark=False, TF32=True. No robot evidence. Each family retains its own normalizer. Local residual once after ensemble/native conversion, before existing robot limit. M5 seed4 fixed before robot trials.\n')
    (root/'manifest.json').write_text(json.dumps(dict(sha256={p.relative_to(root).as_posix():digest(p) for p in root.rglob('*') if p.is_file()}),indent=2))
    for model in MODELS:verify_bundle(root,model)
    archive=WORK/'R5_deploy_bundle.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(root,arcname='R5')
    receipt=dict(status='PREPARED_NOT_PUBLISHED',archive=str(archive),bytes=archive.stat().st_size,sha256=digest(archive),models=identities,robot_trials=0)
    (WORK/'bundle_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
