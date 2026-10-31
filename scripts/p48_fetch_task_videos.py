"""Pin video files for three selected-task dataset sanity episodes; no training."""
import json
from pathlib import Path
import ssl
import time

import httpx
import pandas as pd
from huggingface_hub import snapshot_download,set_client_factory
from huggingface_hub.errors import HfHubHTTPError

OUT=Path('/root/autodl-tmp/lerobot-main/experiments/p4_public_benchmark/p4_8_libero_pilot/index1/result')
DATA=Path('/root/autodl-tmp/p48_data/lerobot_libero')


def main():
    task=json.loads((OUT/'selected_task.json').read_text());ids=task['episode_ids']
    chosen=[ids[0],ids[len(ids)//2],ids[-1]]
    meta=pd.read_parquet(DATA/'meta/episodes/chunk-000/file-000.parquet')
    info=json.loads((DATA/'meta/info.json').read_text());paths=set()
    for _,row in meta[meta.episode_index.isin(chosen)].iterrows():
        for camera in ['observation.images.image','observation.images.image2']:
            paths.add(info['video_path'].format(video_key=camera,chunk_index=int(row[f'videos/{camera}/chunk_index']),file_index=int(row[f'videos/{camera}/file_index'])))
    context=ssl.create_default_context();context.maximum_version=ssl.TLSVersion.TLSv1_2
    set_client_factory(lambda:httpx.Client(verify=context,timeout=httpx.Timeout(120,connect=30)))
    source=json.loads((OUT/'dataset_source.json').read_text())
    for attempt in range(5):
        try:
            snapshot_download(source['repo_id'],revision=source['revision'],repo_type='dataset',local_dir=DATA,token=False,
                              allow_patterns=sorted(paths),max_workers=4);break
        except (httpx.TransportError,HfHubHTTPError) as error:
            status=getattr(getattr(error,'response',None),'status_code',None)
            if status is not None and status!=429 and status<500:raise
            print('VIDEO_RETRY',attempt+1,type(error).__name__,status,flush=True)
            if attempt==4:raise
            time.sleep(5*(attempt+1))
    (OUT/'video_sanity_selection.json').write_text(json.dumps(dict(episode_ids=chosen,files=sorted(paths),
        selection='First/median/last selected-task episode before any model performance; sanity only, not prototype selection'),indent=2))
    print('TASK_SANITY_VIDEOS_READY',chosen,len(paths),flush=True)


if __name__=='__main__':main()
