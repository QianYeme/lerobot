"""Real LeRobotDataset temporal/video-index regression on pinned LIBERO data."""
import json
from pathlib import Path
import sys

OUT=Path('/root/autodl-tmp/lerobot-main/experiments/p4_public_benchmark/p4_8_libero_pilot/index1/result')
DATA=Path('/root/autodl-tmp/p48_data/lerobot_libero')
sys.path.insert(0,str(OUT/'workspace/lerobot/src'))


def main():
    import numpy as np
    import pandas as pd
    import torch
    from torchcodec.decoders import VideoDecoder
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    torch.set_num_threads(4)
    chosen=json.loads((OUT/'video_sanity_selection.json').read_text())['episode_ids']
    revision=json.loads((OUT/'dataset_source.json').read_text())['revision']
    dataset=LeRobotDataset('lerobot/libero',root=DATA,episodes=chosen,revision=revision,
                          delta_timestamps={'action':[i/10 for i in range(100)]},video_backend='torchcodec')
    meta=pd.read_parquet(DATA/'meta/episodes/chunk-000/file-000.parquet')
    offset=0;records=[];frames_dir=OUT/'figures';frames_dir.mkdir(exist_ok=True)
    import imageio.v2 as imageio
    for ep in chosen:
        row=meta[meta.episode_index==ep].iloc[0];length=int(row.length)
        for index in [0,length//2,length-1]:
            raw=dataset[offset+index]
            assert int(raw['episode_index'])==ep and int(raw['frame_index'])==index
            assert raw['observation.state'].shape==(8,) and raw['action'].shape==(100,7)
            assert torch.isfinite(raw['action']).all() and torch.isfinite(raw['observation.state']).all()
            if index==length-1:
                assert int(raw['action_is_pad'].sum())==99
                assert torch.equal(raw['action'],raw['action'][0:1].expand(100,-1))
            differences={};pictures=[]
            for key in ['observation.images.image','observation.images.image2']:
                image=raw[key];assert image.shape==(3,256,256) and image.dtype==torch.float32
                file=DATA/f"videos/{key}/chunk-{int(row[f'videos/{key}/chunk_index']):03d}/file-{int(row[f'videos/{key}/file_index']):03d}.mp4"
                seconds=float(row[f'videos/{key}/from_timestamp'])+index/10+1e-4
                decoder=VideoDecoder(str(file),seek_mode='exact')
                expected=decoder.get_frame_played_at(seconds).data.float()/255
                diff=float((image-expected).abs().max());assert diff<=1e-7
                differences[key]=diff
                pictures.append((image.permute(1,2,0).numpy()*255).round().astype(np.uint8))
            if index==0:imageio.imwrite(frames_dir/f'dataset_ep{ep}_first.png',np.concatenate(pictures,axis=1))
            records.append(dict(episode=ep,frame=index,length=length,action_shape=list(raw['action'].shape),
                action_pad=int(raw['action_is_pad'].sum()),direct_video_pixel_max_difference=differences,
                state=raw['observation.state'].tolist(),h0_action=raw['action'][0].tolist()))
        offset+=length
    (OUT/'dataset_sanity_samples.json').write_text(json.dumps(records,indent=2))
    (OUT/'dataset_probe_complete.json').write_text(json.dumps(dict(status='REAL_VIDEO_STATE_ACTION_PADDING_PASS',
        samples=len(records),episode_ids=chosen,uses_global_normalizers=False,policy_training=False),indent=2))
    print('REAL_DATASET_PROBE_PASS',len(records),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        (OUT/'dataset_probe_error.log').write_text(traceback.format_exc());raise
