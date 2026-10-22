"""Publish frozen F5 artifacts from the server; token arrives via stdin only."""
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

WORK=Path('/root/autodl-tmp/F5_20261022')
REPO='QYyyyyyyy/F5_UNIFIED_SO101'


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    token=sys.stdin.readline().strip()
    if not token:raise ValueError('HF token required on stdin; never save it to disk')
    api=HfApi(endpoint='https://huggingface.co',token=token)
    if api.whoami()['name']!='QYyyyyyyy':raise ValueError('Unexpected publishing account')
    receipt=json.loads((WORK/'bundle_receipt.json').read_text())
    archive=WORK/'F5_deploy_bundle.tar.gz'
    if archive.stat().st_size!=receipt['bytes'] or digest(archive)!=receipt['sha256']:
        raise ValueError('Frozen model archive changed')
    api.create_repo(REPO,repo_type='model',exist_ok=True,private=False)
    paths={
        'README.md':WORK/'HF_MODEL_CARD.md',
        'REAL_ROBOT_TEST_GUIDE.md':WORK/'REAL_ROBOT_TEST_GUIDE.md',
        'F5_deploy_bundle.tar.gz':archive,
        'F5_project_files.tar.gz':WORK/'F5_project_files.tar.gz',
        'model_identity.json':WORK/'bundle/model_identity.json',
        'manifest.json':WORK/'bundle/manifest.json',
        'bundle_receipt.json':WORK/'bundle_receipt.json',
        'runtime_smoke.json':WORK/'runtime_smoke.json'}
    expected={name:dict(bytes=path.stat().st_size,sha256=digest(path)) for name,path in paths.items()}
    for name,path in paths.items():
        print('Uploading '+name,flush=True)
        api.upload_file(path_or_fileobj=path,path_in_repo=name,repo_id=REPO,repo_type='model',
            commit_message='Publish frozen F5 deployment artifact: '+name)
    info=api.model_info(REPO,files_metadata=True)
    files={entry.rfilename:entry for entry in info.siblings}
    for name,values in expected.items():
        entry=files[name]
        if entry.size!=values['bytes']:raise ValueError('Remote size mismatch: '+name)
        if entry.lfs:
            if entry.lfs.sha256!=values['sha256']:raise ValueError('Remote LFS SHA mismatch: '+name)
        else:
            # Git blob identity verifies small files without downloading model payloads.
            payload=paths[name].read_bytes()
            blob=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
            if entry.blob_id!=blob:raise ValueError('Remote Git blob mismatch: '+name)
    result=dict(status='PUBLISHED_VERIFIED',repo=REPO,revision=info.sha,files=expected,
        verification='Remote byte sizes and LFS SHA256 / Git blob SHA1',robot_trials=0,F5_COMPLETE=False)
    (WORK/'hf_publish_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
