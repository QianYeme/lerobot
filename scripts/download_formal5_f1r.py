"""Download immutable Formal5 F1R sources, without assigning experimental conditions."""
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPOS = ['QYyyyyyyy/formal5_kind1', 'QYyyyyyyy/formal5_kind2']


def main():
    api = HfApi()
    records = []
    for repo in REPOS:
        info = api.dataset_info(repo, files_metadata=True)
        records.append({'repo_id': repo, 'revision': info.sha,
                        'destination': str(ROOT / '数据集/formal5' / repo.split('/')[-1]),
                        'files': [{'path': x.rfilename, 'bytes': x.size} for x in info.siblings],
                        'condition': 'UNASSIGNED_PENDING_OPERATOR_CONFIRMATION'})
    raw = ROOT / 'results/experiments/F1R_ACTUATOR_LOAD_CONTACT_INTERVENTION' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    raw.mkdir(parents=True)
    manifest = {'sources': records, 'raw': str(raw), 'datasets_modified': False}
    (raw / 'download_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    def download(r):
        snapshot_download(repo_id=r['repo_id'], repo_type='dataset', revision=r['revision'],
                          local_dir=r['destination'], max_workers=3)
        assert all((Path(r['destination']) / x['path']).is_file() for x in r['files'])
        print(r['repo_id'], r['revision'], 'downloaded', flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download, records))
    print(str(raw), flush=True)


if __name__ == '__main__':
    main()
