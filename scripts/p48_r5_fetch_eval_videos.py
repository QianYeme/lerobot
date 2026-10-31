"""Four-way public HTTPS video fetch with fresh redirects and pinned LFS checksums."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time

import pandas as pd

OUT=Path('/root/autodl-tmp/lerobot-main/experiments/p4_public_benchmark/p4_8_libero_pilot/index5/result')
DATA=Path('/root/autodl-tmp/p48_data/lerobot_libero')


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    source=json.loads((OUT.parent.parent/'index1/result/dataset_source.json').read_text());task=json.loads((OUT/'dataset_split_manifest.json').read_text())
    chosen=task['dev_episode_ids']
    meta=pd.read_parquet(DATA/'meta/episodes/chunk-000/file-000.parquet');info=json.loads((DATA/'meta/info.json').read_text());paths=set()
    for _,row in meta[meta.episode_index.isin(chosen)].iterrows():
        for camera in ['observation.images.image','observation.images.image2']:
            paths.add(info['video_path'].format(video_key=camera,chunk_index=int(row[f'videos/{camera}/chunk_index']),file_index=int(row[f'videos/{camera}/file_index'])))
    metadata=json.loads((OUT/'pinned_hf_file_metadata.json').read_text())
    assert metadata['revision']==source['revision']
    expected={entry['rfilename']:entry for entry in metadata['siblings']}
    def fetch(name):
        file=DATA/name;file.parent.mkdir(parents=True,exist_ok=True);entry=expected[name]
        if file.exists() and file.stat().st_size==entry['size'] and sha(file)==entry['lfs']['sha256']:
            return dict(path=name,bytes=entry['size'],sha256=entry['lfs']['sha256'],status='EXISTING_VERIFIED')
        temp=file.with_suffix('.http_partial')
        for attempt in range(4):
            url=f"https://huggingface.co/datasets/{source['repo_id']}/resolve/{source['revision']}/{name}?download=true&p48nocache={time.time_ns()}"
            process=subprocess.run(['curl','--tls-max','1.2','--fail','--location','--silent','--show-error',
                '--connect-timeout','15','--max-time','240','--output',str(temp),url],capture_output=True,text=True)
            if process.returncode==0 and temp.stat().st_size==entry['size'] and sha(temp)==entry['lfs']['sha256']:
                temp.replace(file);print('VIDEO_SHA_PASS',name,flush=True)
                return dict(path=name,bytes=entry['size'],sha256=entry['lfs']['sha256'],status='DOWNLOADED_VERIFIED')
            print('PUBLIC_HTTPS_RETRY',name,attempt+1,process.returncode,flush=True);time.sleep(2*(attempt+1))
        raise RuntimeError('Pinned video HTTP fetch failed: '+name)
    with ThreadPoolExecutor(max_workers=4) as executor:checks=list(executor.map(fetch,sorted(paths)))
    (OUT/'video_download_checksums.json').write_text(json.dumps(checks,indent=2))
    (OUT/'video_sanity_selection.json').write_text(json.dumps(dict(episode_ids=chosen,files=sorted(paths),
        selection='Frozen Dev9 video chunks, fetched only after prototype freeze; not for prototype selection'),indent=2))
    print('DEV9_VIDEOS_READY',chosen,len(paths),flush=True)


if __name__=='__main__':main()
