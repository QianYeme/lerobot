"""Publish the verified R5.11 historical133k deployment bundle; token is environment-only."""
import hashlib
import json
import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi

W=Path('/root/autodl-tmp/R5_11_20261025')
REPO='QYyyyyyyy/R5_11_FULL60_133K_SO101'


def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    smoke=json.loads((W/'runtime_smoke.json').read_text())
    assert smoke['status']=='PASS_OFFLINE_ONLY' and smoke['samples']==36
    publish=W/'publish'
    if not publish.exists():shutil.copytree(W/'bundle',publish,copy_function=os.link)
    stage=W/'source/experiments/force/r5_11_real_robot_133k_breakthrough_diagnostic/index1'
    for p in stage.iterdir():
        if p.is_file() and p.suffix in ['.md','.json','.csv']:shutil.copy2(p,publish/p.name)
    for name in ['R5_11_project_files.tar.gz','runtime_smoke.json']:
        shutil.copy2(W/name,publish/name)
    (publish/'README.md').write_text('---\ntags:\n- lerobot\n- robotics\n- act\n---\n# R5.11 historical Full60/seed3/133k\n\nOnly R3 from the historical R5.3 experiment is included. No reference-model weights are uploaded.\nGlobal-only G-Load, 6D joint state plus independent single-frame load, 6D actions, Full60 normalization.\nOriginal weight SHA256: d094de659f58df3e55cd6284a83729cfd8658fb1d4b19f73d6a8cb634a61fc0f.\nThis is not a new deterministic run, clip5, SWA, or a continuation of R5.10 M1.\nSee REAL_ROBOT_TEST_GUIDE.md. Use the included source archive for the matching new entry.\nManual reset; no numeric Home/tolerance file, Stage0, shadow or persistent STOP gate.\nPhysical limits and human stop remain. Parity evidence uses training inputs and is not validation or robot success.\nStage A first; Stage B conditional on an observed and confirmed robot signal. No robot trials performed by this release.\nR5_11_COMPLETE=false; Builder/Red-Team review pending.\n',encoding='utf-8')
    expected={p.relative_to(publish).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p)) for p in publish.rglob('*') if p.is_file() and p.name!='checksums.json' and '.cache' not in p.parts}
    (publish/'checksums.json').write_text(json.dumps(expected,indent=2)+'\n')
    expected['checksums.json']=dict(bytes=(publish/'checksums.json').stat().st_size,sha256=digest(publish/'checksums.json'))
    api=HfApi(endpoint='https://huggingface.co',token=os.environ['HF_TOKEN'])
    assert api.whoami()['name']=='QYyyyyyyy'
    print('ACCOUNT_VERIFIED; uploading',len(expected),'files',flush=True)
    api.create_repo(REPO,repo_type='model',private=False,exist_ok=True)
    commit=api.upload_folder(folder_path=publish,repo_id=REPO,repo_type='model',ignore_patterns=['.cache/**'],commit_message='Publish only original R5.3 133k model and R5.11 direct-run guide')
    info=api.model_info(REPO,revision=commit.oid,files_metadata=True)
    remote={p.rfilename:p for p in info.siblings}
    for name,value in expected.items():
        entry=remote[name];assert entry.size==value['bytes'],name
        if entry.lfs:assert entry.lfs.sha256==value['sha256'],name
        else:
            raw=(publish/name).read_bytes()
            assert entry.blob_id==hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),name
    receipt=dict(status='PUBLISHED_VERIFIED',repo=REPO,revision=info.sha,files=expected,robot_trials=0,R5_11_COMPLETE=False,Home='MANUAL_RESET_NO_NUMERIC_GATE',verification='Pinned commit: every file size and LFS SHA256 or Git blob SHA1')
    (W/'hf_publish_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('PUBLISHED_VERIFIED',REPO,info.sha,flush=True)


if __name__=='__main__':main()
