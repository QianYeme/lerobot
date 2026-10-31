"""Download exact P4.7 private revisions using the operator's HF login; verify every file."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--receipts',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path('outputs/deploy/P47'))
    p.add_argument('--models',nargs='*')
    args=p.parse_args()
    for item in json.loads(args.receipts.read_text()):
        if args.models and item['role'] not in args.models:continue
        root=args.root/item['role']
        snapshot_download(repo_id=item['repo_id'],revision=item['revision'],local_dir=root)
        for entry in json.loads((root/'checksums.json').read_text()):
            path=(root/entry['path']).resolve();assert path.is_relative_to(root.resolve())
            with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            assert actual==entry['sha256'],entry['path']
        (root/'download_receipt.json').write_text(json.dumps(dict(repo_id=item['repo_id'],revision=item['revision'],all_sha_pass=True),indent=2))
        print(item['role'],'DOWNLOAD_SHA_PASS')


if __name__=='__main__':main()
