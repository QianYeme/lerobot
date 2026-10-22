"""Prepare an unpublished F5 deployment bundle from exact frozen server artifacts."""
import json
import os
import shutil
import tarfile
from pathlib import Path

from lerobot.utils.f5_protocol import MODELS, EXPECTED, POSITION_FIELDS, BOUND, digest, verify_bundle

WORK=Path('/root/autodl-tmp/F5_20261022')
SOURCE=Path('/root/autodl-tmp/F3_r1_transfer_20261020/bundle')


def main():
    if shutil.disk_usage('/root/autodl-tmp').free<18*1024**3:raise RuntimeError('Preserve15GiB disk reserve')
    WORK.mkdir(exist_ok=True);root=WORK/'bundle';root.mkdir(exist_ok=False)
    paths={
        'P1-Hist':SOURCE/'models/M0',
        'Strong-G':Path('/root/autodl-tmp/F4_1_20261022/runs/G-Load/seed3/attempt1/checkpoints/080000/pretrained_model'),
        'B0-6D':SOURCE/'global'}
    for name,path in paths.items():shutil.copytree(path,root/'models'/name,copy_function=os.link)
    (root/'local').mkdir()
    locals={
        'L-Load':Path('/root/autodl-tmp/F4_1_20261022/runs/L-Load/seed4/attempt1/009000.pt'),
        'New-GL':Path('/root/autodl-tmp/F4_2_20261022/result/New-GL/seed3/007000.pt')}
    for name,path in locals.items():os.link(path,root/'local'/f'{name}.pt')
    shutil.copytree(SOURCE/'assets',root/'assets',copy_function=os.link)
    shutil.copytree(SOURCE/'torch_cache',root/'torch_cache',copy_function=os.link,ignore=shutil.ignore_patterns('__pycache__','.git'))
    shutil.copy2(SOURCE/'normalization.json',root/'normalization.json')
    identities={}
    for name in MODELS:
        global_name='B0-6D' if name=='L-Load' else ('Strong-G' if name=='New-GL' else name)
        gsha,lsha=EXPECTED[name]
        identity=dict(global_path='models/'+global_name,global_sha256=gsha,
            state_fields=POSITION_FIELDS+(['gripper.load','gripper.curr'] if name=='P1-Hist' else []),
            global_load=name in ['Strong-G','New-GL'],role='historical practical reference' if name=='P1-Hist' else 'F5 candidate')
        if lsha:identity.update(local_path=f'local/{name}.pt',local_sha256=lsha,bound=BOUND)
        identities[name]=identity
    (root/'model_identity.json').write_text(json.dumps(identities,indent=2))
    manifest={p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob('*')) if p.is_file()}
    (root/'manifest.json').write_text(json.dumps(dict(sha256=manifest),indent=2))
    for model in MODELS:verify_bundle(root,model)
    (root/'README.md').write_text('F5 exact four-model deployment bundle. No trial results or deployment approval. Use project control_f5.py and F5 REAL_ROBOT_TEST_GUIDE.md. Keep torch_cache for offline startup. Local residual is applied exactly once after global temporal ensemble/denormalization, before robot relative limiting.\n')
    archive=WORK/'F5_deploy_bundle.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(root,arcname='F5')
    receipt=dict(status='BUNDLE_PREPARED_NOT_PUBLISHED',bundle=str(root),archive=str(archive),bytes=archive.stat().st_size,
        sha256=digest(archive),models=MODELS,hardware='NOT_CONNECTED',final12_reads=0)
    (WORK/'bundle_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
