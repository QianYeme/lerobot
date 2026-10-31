"""Add final guide, no-action diagnostics and exact download receipts to private releases."""
import json
from pathlib import Path
import re
import shutil
import ssl
import subprocess
import sys
import time

import httpx
from huggingface_hub import HfApi,set_client_factory,snapshot_download
from p47_stage_models import WORK,ROOT,sha
from p47_publish import files


def main(token):
    context=ssl.create_default_context();context.maximum_version=ssl.TLSVersion.TLSv1_2
    set_client_factory(lambda:httpx.Client(verify=context,timeout=httpx.Timeout(180,connect=30)))
    api=HfApi(token=token)
    out=WORK/'result/upload';receipts=json.loads((out/'upload_receipts.json').read_text());assert len(receipts)==7
    for item in receipts:
        if item.get('final_all_file_sha_verified'):continue
        root=WORK/'staging'/item['role'];fresh=WORK/'download_verify'/item['role']
        for source,target in [(ROOT/'scripts/p47_shadow.py',root/'p47_shadow.py'),
                              (WORK/'REAL_ROBOT_TEST_GUIDE.md',root/'REAL_ROBOT_TEST_GUIDE.md')]:shutil.copy2(source,target)
        record=json.loads((root/'manifest.json').read_text())
        record.update(upload_status='PRIVATE_RELEASE',validated_prior_revision=item['revision'],
                      hardware_validation='PENDING_OPERATOR',git_commit_scope='Server checkout; exact deployment source content hashes are in checksums.json')
        (root/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
        checks=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files(root) if p.name!='checksums.json']
        (root/'checksums.json').write_text(json.dumps(checks,indent=2)+'\n')
        additions=['p47_shadow.py','REAL_ROBOT_TEST_GUIDE.md','manifest.json','checksums.json']
        for name in additions:
            assert not re.search(r'hf_[A-Za-z0-9]{20,}',(root/name).read_text())
        from huggingface_hub import CommitOperationAdd
        commit=api.create_commit(item['repo_id'],operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(root/name)) for name in additions],
                                 commit_message='Finalize P4.7 guide, manifests and no-action object diagnostics')
        # Download only changed docs; all weight/source files stay pinned by checksum.
        snapshot_download(item['repo_id'],revision=commit.oid,local_dir=fresh,token=api.token,
                          allow_patterns=additions+['runtime_validation.json'],force_download=True,max_workers=2)
        for entry in checks:assert sha(fresh/entry['path'])==entry['sha256']
        assert api.model_info(item['repo_id'],revision=commit.oid).private
        item.update(verified_model_revision=item['revision'],revision=commit.oid,final_all_file_sha_verified=True,
                    supplemental_diagnostics_validation='AST_NO_SEND_ACTION; HARDWARE_PENDING')
        (out/'upload_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
        print('FINAL_PRIVATE_RELEASE_PASS',item['role'],commit.oid,flush=True)


if __name__=='__main__':
    from huggingface_hub.errors import HfHubHTTPError
    token=sys.stdin.readline().strip()
    for attempt in range(5):
        try:
            main(token);break
        except (HfHubHTTPError,httpx.TransportError) as error:
            status=getattr(getattr(error,'response',None),'status_code',None)
            if status is not None and status!=429 and status<500:raise
            print('FINAL_DOCS_NETWORK_RETRY',attempt+1,type(error).__name__,status,flush=True)
            if attempt==4:raise
            time.sleep(10*(attempt+1))
