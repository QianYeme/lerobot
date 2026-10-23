"""Publish verified R5 files from server; credentials come only from environment."""
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

WORK=Path('/root/autodl-tmp/R5_20261023')
REPO='QYyyyyyyy/R5_PRE_FINAL_DATASET_SO101'


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    smoke=json.loads((WORK/'runtime_smoke.json').read_text())
    assert smoke['status']=='PASS_OFFLINE_ONLY' and len(smoke['models'])==5
    receipt=json.loads((WORK/'bundle_receipt.json').read_text())
    archive=WORK/'R5_deploy_bundle.tar.gz'
    assert digest(archive)==receipt['sha256'] and archive.stat().st_size==receipt['bytes']
    api=HfApi(endpoint='https://huggingface.co',token=os.environ['HF_TOKEN'])
    assert api.whoami()['name']=='QYyyyyyyy','Wrong account'
    publish=WORK/'publish';publish.mkdir(exist_ok=False)
    files={'README.md':WORK/'HF_MODEL_CARD.md','REAL_ROBOT_TEST_GUIDE.md':WORK/'REAL_ROBOT_TEST_GUIDE.md',
        'R5_deploy_bundle.tar.gz':archive,'R5_project_files.tar.gz':WORK/'R5_project_files.tar.gz',
        'model_identity.json':WORK/'bundle/model_identity.json','manifest.json':WORK/'bundle/manifest.json',
        'runtime_smoke.json':WORK/'runtime_smoke.json','bundle_receipt.json':WORK/'bundle_receipt.json',
        'decision_R5.md':WORK/'decision.md','bad_seed_selection.md':WORK/'bad_seed_selection.md'}
    expected={name:dict(bytes=path.stat().st_size,sha256=digest(path)) for name,path in files.items()}
    for name,path in files.items():os.link(path,publish/name)
    (publish/'checksums.json').write_text(json.dumps(expected,indent=2)+'\n')
    expected['checksums.json']=dict(bytes=(publish/'checksums.json').stat().st_size,sha256=digest(publish/'checksums.json'))
    api.create_repo(REPO,repo_type='model',private=False,exist_ok=True)
    commit=api.upload_folder(folder_path=publish,repo_id=REPO,repo_type='model',commit_message='Publish R5 five frozen SO101 real-robot gate models and guide')
    info=api.model_info(REPO,revision=commit.oid,files_metadata=True)
    remote={entry.rfilename:entry for entry in info.siblings}
    for name,values in expected.items():
        entry=remote[name];assert entry.size==values['bytes'],name
        if entry.lfs:assert entry.lfs.sha256==values['sha256'],name
        else:
            payload=(publish/name).read_bytes()
            assert entry.blob_id==hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest(),name
    result=dict(status='PUBLISHED_VERIFIED',repo=REPO,revision=info.sha,files=expected,
        verification='Remote sizes and LFS SHA256/Git blob SHA1; pinned upload commit',robot_trials=0,R5_COMPLETE=False)
    (WORK/'hf_publish_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
