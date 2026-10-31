"""Fetch pinned public metadata/state/action tables only; videos follow task matching."""
import fcntl
import json
from pathlib import Path
import ssl
import time

import httpx
from huggingface_hub import snapshot_download,set_client_factory

OUT=Path('/root/autodl-tmp/lerobot-main/experiments/p4_public_benchmark/p4_8_libero_pilot/index1/result')
DATA=Path('/root/autodl-tmp/p48_data/lerobot_libero')


def main():
    lock=(OUT/'data_fetch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=json.loads((OUT/'dataset_source.json').read_text())
    ctx=ssl.create_default_context();ctx.maximum_version=ssl.TLSVersion.TLSv1_2
    set_client_factory(lambda:httpx.Client(verify=ctx,timeout=httpx.Timeout(120,connect=30)))
    for attempt in range(5):
        try:
            snapshot_download('lerobot/libero',repo_type='dataset',revision=source['revision'],local_dir=DATA,
                              token=False,allow_patterns=['meta/**','data/**'],max_workers=8)
            break
        except httpx.TransportError as error:
            print('PUBLIC_DATA_NETWORK_RETRY',attempt+1,type(error).__name__,flush=True)
            if attempt==4:raise
            time.sleep(5*(attempt+1))
    import pandas as pd
    table=pd.concat([pd.read_parquet(p) for p in sorted((DATA/'data').rglob('*.parquet'))],ignore_index=True)
    info=json.loads((DATA/'meta/info.json').read_text())
    assert len(table)==info['total_frames']
    counts=table.groupby('task_index').agg(episodes=('episode_index','nunique'),frames=('frame_index','size')).reset_index()
    tasks=pd.read_parquet(DATA/'meta/tasks.parquet').reset_index().rename(columns={'index':'instruction'})
    merged=counts.merge(tasks,on='task_index',validate='one_to_one')
    (OUT/'dataset_task_counts.json').write_text(merged.to_json(orient='records',indent=2))
    (OUT/'data_fetch_complete.json').write_text(json.dumps(dict(status='TABLES_READY',revision=source['revision'],frames=len(table),tasks=len(merged),data_root=str(DATA),videos_downloaded=False,training=False),indent=2))
    print('PINNED_TABLES_READY',len(table),len(merged),flush=True)


if __name__=='__main__':main()
