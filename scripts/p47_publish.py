"""Publish private deployment repos; credential read once from stdin, never persisted."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import logging
import ssl
import httpx
import time

from huggingface_hub import HfApi,snapshot_download,hf_hub_download,CommitOperationCopy
from p47_stage_models import WORK,ROOT,sha


def files(root):
    return [p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.cache' not in p.parts and p.suffix!='.pyc']


def main(token=None):
    from huggingface_hub import set_client_factory
    context=ssl.create_default_context();context.maximum_version=ssl.TLSVersion.TLSv1_2
    set_client_factory(lambda:httpx.Client(verify=context,timeout=httpx.Timeout(180,connect=30)))
    logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
    token=token or sys.stdin.readline().strip();assert token
    api=HfApi(token=token);assert api.whoami()['name']=='QYyyyyyyy'
    donor='QYyyyyyyy/P4_3_FORMAL6_SO101'
    donor_info=api.model_info(donor,revision='a0e55ec840709cf0cacb59ccb26a3e36dffd11ec',files_metadata=True)
    reusable={entry.lfs.sha256:entry.rfilename for entry in donor_info.siblings if entry.lfs}
    out=WORK/'result/upload'
    receipts=json.loads((out/'upload_receipts.json').read_text()) if (out/'upload_receipts.json').exists() else []
    existing={item.id for item in api.list_models(author='QYyyyyyyy')}
    inventory=json.loads((out/'model_inventory.json').read_text())
    mapping=[]
    if (out/'repo_mapping.json').exists():mapping=json.loads((out/'repo_mapping.json').read_text())
    else:
        for item in inventory:
            repo=item['proposed_repo_id']
            if repo in existing:repo+='-p47-20261030'
            if repo in existing:raise RuntimeError('Existing destination requires explicit identity audit; no overwrite')
            mapping.append(dict(role=item['role'],repo_id=repo))
    (out/'repo_mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    for item,mapped in zip(inventory,mapping):
        role=item['role'];root=WORK/'staging'/role;repo=mapped['repo_id']
        if any(r['role']==role and r['repo_id']==repo for r in receipts):continue
        validation=json.loads((root/'runtime_validation.json').read_text());assert validation['status']=='PASS'
        smoke=out/(role+'_control_smoke.json');assert smoke.exists()
        manifest=json.loads((root/'manifest.json').read_text())
        manifest.update(repo_id=repo,upload_status='PREPARED_FOR_PRIVATE_RELEASE',inference_validation=validation)
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (root/'requirements-server.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        checks=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files(root) if p.name!='checksums.json']
        (root/'checksums.json').write_text(json.dumps(checks,indent=2)+'\n')
        for path in files(root):
            if path.suffix in ['.py','.md','.json','.txt','.toml','.yaml','.yml']:
                if re.search(r'hf_[A-Za-z0-9]{20,}',path.read_text(errors='replace')):
                    raise RuntimeError('Credential pattern detected in '+str(path.relative_to(root)))
        if api.repo_exists(repo,repo_type='model'):
            info=api.model_info(repo);assert info.private
            if any(p.rfilename=='manifest.json' for p in info.siblings):
                path=hf_hub_download(repo,'manifest.json',token=token)
                assert json.loads(Path(path).read_text())['source_checkpoint_sha256']==manifest['source_checkpoint_sha256']
            else:assert all(p.rfilename in {entry['path'] for entry in checks}|{'.gitattributes'} for p in info.siblings)
        else:api.create_repo(repo,private=True,repo_type='model',exist_ok=False)
        copies=[entry for entry in checks if entry['sha256'] in reusable]
        if copies:
            from huggingface_hub.errors import FileDuplicationError
            try:
                api.create_commit(repo_id=repo,operations=[CommitOperationCopy(src_path_in_repo=reusable[entry['sha256']],
                    path_in_repo=entry['path'],src_repo_id=donor,src_repo_type='model',src_revision=donor_info.sha) for entry in copies],
                    commit_message='Reuse hash-identical verified inference assets from P4.3')
            except FileDuplicationError:
                # Earlier LFS attempts may already have stored the same objects without Xet metadata.
                # The supported upload API checks existing objects and resumes without replacing them.
                print('COPY_NOT_SUPPORTED_FOR_EXISTING_LFS_OBJECTS',role,flush=True)
                copies=[]
        print('SERVER_SIDE_HASH_IDENTICAL_COPIES',role,len(copies),flush=True)
        commit=api.upload_folder(repo_id=repo,folder_path=str(root),ignore_patterns=['**/__pycache__/**','**/*.pyc','.cache/**']+[entry['path'] for entry in copies],commit_message='P4.7 verified deployment-only private release')
        info=api.model_info(repo,revision=commit.oid,files_metadata=True);assert info.private
        remote={p.rfilename:p for p in info.siblings}
        for entry in checks:
            obj=remote[entry['path']];assert obj.size==entry['bytes']
            if obj.lfs:assert obj.lfs.sha256==entry['sha256']
        print('PRIVATE_UPLOAD_COMMITTED',role,repo,commit.oid,flush=True)
        fresh=WORK/'download_verify'/role
        for attempt in range(5):
            try:
                snapshot_download(repo_id=repo,revision=commit.oid,local_dir=fresh,token=token,max_workers=4)
                break
            except httpx.TransportError as error:
                print('DOWNLOAD_TRANSPORT_RETRY',role,attempt+1,type(error).__name__,flush=True)
                if attempt==4:raise
                time.sleep(5*(attempt+1))
        for entry in checks:assert sha(fresh/entry['path'])==entry['sha256']
        process=subprocess.run([sys.executable,str(ROOT/'scripts/p47_validate_package.py'),'--bundle',str(fresh)],capture_output=True,text=True)
        (out/(role+'_download_validation.log')).write_text(process.stdout+process.stderr)
        assert process.returncode==0,'Independent download load/equivalence failed: '+role
        receipt=dict(role=role,repo_id=repo,revision=commit.oid,private=info.private,files=len(checks)+1,
                     sizes_lfs_sha_verified=True,all_download_sha_verified=True,download_runtime_validation='PASS',
                     download_location=str(fresh),local_control_host_validation='PENDING_OPERATOR_MACHINE')
        receipts.append(receipt);(out/'upload_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
        print('PRIVATE_UPLOAD_AND_DOWNLOAD_PASS',role,repo,commit.oid,flush=True)


if __name__=='__main__':
    from huggingface_hub.errors import HfHubHTTPError
    credential=sys.stdin.readline().strip()
    for attempt in range(5):
        try:
            main(credential)
            break
        except (HfHubHTTPError,httpx.TransportError) as error:
            status=getattr(getattr(error,'response',None),'status_code',None)
            if status is not None and status!=429 and status<500:raise
            print('RELEASE_NETWORK_RETRY',attempt+1,type(error).__name__,status,flush=True)
            if attempt==4:raise
            time.sleep(10*(attempt+1))
