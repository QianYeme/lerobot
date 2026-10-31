"""Pinned simulator assets only; no policies, demonstrations or credentials."""
import fcntl
import hashlib
import json
from pathlib import Path
import ssl
import time

import httpx
from huggingface_hub import snapshot_download,set_client_factory
from huggingface_hub.errors import HfHubHTTPError

OUT=Path('/root/autodl-tmp/lerobot-main/experiments/p4_public_benchmark/p4_8_libero_pilot/index1/result')
DEST=Path('/root/autodl-tmp/p48_data/libero_assets')


def main():
    lock=(OUT/'assets_fetch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=json.loads((OUT/'assets_source.json').read_text())
    context=ssl.create_default_context();context.maximum_version=ssl.TLSVersion.TLSv1_2
    set_client_factory(lambda:httpx.Client(verify=context,timeout=httpx.Timeout(120,connect=30)))
    for attempt in range(5):
        try:
            snapshot_download(source['repo_id'],revision=source['revision'],repo_type='dataset',local_dir=DEST,token=False,max_workers=12)
            break
        except (httpx.TransportError,HfHubHTTPError) as error:
            status=getattr(getattr(error,'response',None),'status_code',None)
            if status is not None and status!=429 and status<500:raise
            print('ASSETS_NETWORK_RETRY',attempt+1,type(error).__name__,status,flush=True)
            if attempt==4:raise
            time.sleep(5*(attempt+1))
    checks=[]
    for entry in source['files']:
        p=DEST/entry['rfilename'];assert p.stat().st_size==entry['size']
        with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if entry.get('lfs'):assert digest==entry['lfs']['sha256']
        checks.append(dict(path=entry['rfilename'],bytes=p.stat().st_size,sha256=digest))
    (OUT/'assets_checksums.json').write_text(json.dumps(checks,indent=2))
    (OUT/'assets_fetch_complete.json').write_text(json.dumps(dict(status='PINNED_ASSETS_READY',revision=source['revision'],path=str(DEST),files=len(checks)),indent=2))
    print('PINNED_ASSETS_READY',len(checks),flush=True)


if __name__=='__main__':main()
