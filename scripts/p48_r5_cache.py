"""Exact native-orientation RGB cache, with source video/index regressions."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path

from p48_r5_common import DATA,OUT,sha,write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['train','dev'],default='train')
    args=parser.parse_args()
    import av
    import numpy as np
    import pandas as pd
    import torch
    from torchcodec.decoders import VideoDecoder
    split=json.loads((OUT/'dataset_split_manifest.json').read_text())
    meta=pd.read_parquet(DATA/'meta/episodes/chunk-000/file-000.parquet')
    ids=sorted(split[args.split+'_episode_ids'])
    frames=[]
    for file in sorted((DATA/'data').rglob('*.parquet')):
        frame=pd.read_parquet(file,filters=[('episode_index','in',ids)])
        if len(frame):frames.append(frame)
    table=pd.concat(frames).sort_values(['episode_index','frame_index']).reset_index(drop=True)
    assert len(table)==(5476 if args.split=='train' else 1391) and set(table.task_index)=={24} and set(table.episode_index)==set(ids)
    folder=OUT/'data_cache'/args.split;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        completed=json.loads((folder/'complete.json').read_text())
        assert sha(folder/'images.npy')==completed['images_sha256']
        assert sha(folder/'arrays.npz')==completed['arrays_sha256'];return
    target=folder/'images.partial.npy'
    images=np.lib.format.open_memmap(target,mode='w+',dtype=np.uint8,shape=(len(table),2,3,256,256))
    jobs={};end=np.empty(len(table),dtype=np.int64)
    for ep in ids:
        row_indices=np.flatnonzero(table.episode_index.to_numpy()==ep)
        assert np.array_equal(table.frame_index.to_numpy()[row_indices],np.arange(len(row_indices)))
        end[row_indices]=row_indices[-1]
        m=meta.loc[meta.episode_index==ep].iloc[0];assert len(row_indices)==int(m.length)
        for camera,key in enumerate(['observation.images.image','observation.images.image2']):
            prefix='videos/'+key+'/'
            file=DATA/('videos/'+key)/f'chunk-{int(m[prefix+"chunk_index"]):03d}'/f'file-{int(m[prefix+"file_index"]):03d}.mp4'
            start=round(float(m[prefix+'from_timestamp'])*10)
            targets=jobs.setdefault((str(file),camera),{})
            for offset,destination in enumerate(row_indices):
                assert start+offset not in targets
                targets[start+offset]=int(destination)
    def decode(job):
        (name,camera),targets=job;count=0
        with av.open(name) as container:
            container.streams.video[0].thread_type='AUTO'
            container.streams.video[0].codec_context.thread_count=2
            for index,frame in enumerate(container.decode(video=0)):
                if index in targets:
                    rgb=frame.to_ndarray(format='rgb24')
                    images[targets[index],camera]=rgb.transpose(2,0,1);count+=1
                if index>=max(targets):break
        assert count==len(targets),(name,count,len(targets))
        print('RGB_CACHE_VIDEO_DONE',camera,Path(name).name,count,flush=True)
        return count
    with ThreadPoolExecutor(max_workers=6) as executor:counts=list(executor.map(decode,jobs.items()))
    assert sum(counts)==len(table)*2
    images.flush();del images;target.replace(folder/'images.npy')
    np.savez(folder/'arrays.npz',state=np.stack(table['observation.state']).astype('float32'),
             action=np.stack(table.action).astype('float32'),episode=table.episode_index.to_numpy(),
             frame=table.frame_index.to_numpy(),last_index=end)
    images=np.load(folder/'images.npy',mmap_mode='r')
    regressions=[]
    # Pin both source frame lookup and channel/orientation convention against TorchCodec.
    for ep in [ids[0],ids[len(ids)//2],ids[-1]]:
        m=meta.loc[meta.episode_index==ep].iloc[0]
        for frame in [0,int(m.length)//2,int(m.length)-1]:
            idx=int(table.index[(table.episode_index==ep)&(table.frame_index==frame)][0])
            for camera,key in enumerate(['observation.images.image','observation.images.image2']):
                prefix='videos/'+key+'/'
                path=DATA/('videos/'+key)/f'chunk-{int(m[prefix+"chunk_index"]):03d}'/f'file-{int(m[prefix+"file_index"]):03d}.mp4'
                decoder=VideoDecoder(str(path),seek_mode='exact')
                expected=decoder.get_frame_played_at(float(m[prefix+'from_timestamp'])+frame/10+1e-4).data.numpy()
                difference=int(np.abs(images[idx,camera].astype(int)-expected.astype(int)).max())
                assert difference==0
                regressions.append(dict(episode=ep,frame=frame,camera=key,max_uint8_difference=difference))
    if args.split=='train':
        stats={}
        for key in ['observation.state','action']:
            values=np.stack(table[key]);assert np.isfinite(values).all()
            stats[key]=dict(mean=torch.tensor(values.mean(0)),std=torch.tensor(values.std(0).clip(1e-6)))
        for key in ['observation.images.image','observation.images.image2']:
            stats[key]=dict(mean=torch.tensor([.485,.456,.406]).view(3,1,1),std=torch.tensor([.229,.224,.225]).view(3,1,1))
        torch.save(stats,OUT/'train35_statistics.pt')
    write(folder/'complete.json',dict(status='EXACT_'+args.split.upper()+'_RGB_CACHE_PASS',frames=len(table),
        images_sha256=sha(folder/'images.npy'),arrays_sha256=sha(folder/'arrays.npz'),
        source_revision=split['revision'],split_sha256=sha(OUT/'dataset_split_manifest.json'),
        cameras=['observation.images.image','observation.images.image2'],train_image_flip=False,
        source_video_regressions=regressions,statistics_frames=len(table) if args.split=='train' else 0))
    print(args.split.upper()+'_RGB_CACHE_PASS',len(table),flush=True)


if __name__=='__main__':main()
