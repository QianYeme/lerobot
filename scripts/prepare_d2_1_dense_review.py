"""Decode each permitted episode continuously; save clips and every-frame sheets."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parents[1]
dataset=root/'数据集/formal4/kind_merged_nomaster'
out=root/'results/experiments/D2_1_ACTION_CHAIN_EVIDENCE_COMPLETION/dense_review'
out.mkdir(parents=True,exist_ok=True)
split=json.loads((root/'experiments/data/d1_training_readiness/index1/result/split_manifest_v1.json').read_text())
allowed=sorted([e for e in split['episodes'] if e['split'] in ('train','dev')],key=lambda x:(x['split']!='dev',x['episode_index']))
meta=pd.read_parquet(dataset/'meta/episodes/chunk-000/file-000.parquet',filters=[('episode_index','in',[e['episode_index'] for e in allowed])])
manifest=[]
def decode_episode(e):
    ep=e['episode_index'];row=meta[meta.episode_index==ep].iloc[0]
    caps=[];offsets=[]
    for key in ['observation.images.top','observation.images.gripper']:
        p=dataset/'videos'/key/f"chunk-{int(row['videos/'+key+'/chunk_index']):03d}"/f"file-{int(row['videos/'+key+'/file_index']):03d}.mp4"
        cap=cv2.VideoCapture(str(p));offset=round(float(row['videos/'+key+'/from_timestamp'])*30)
        cap.set(cv2.CAP_PROP_POS_FRAMES,offset);caps.append(cap);offsets.append(offset)
    epout=out/f'ep{ep:03d}';epout.mkdir(exist_ok=True)
    writer=cv2.VideoWriter(str(epout/'dual_view.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,480))
    pages=[];sheet=None;decoded=0
    for f in range(int(row.length)):
        images=[]
        for cap in caps:
            ok,img=cap.read()
            if not ok:raise RuntimeError(f'episode {ep} frame {f} decode failed')
            images.append(img)
        writer.write(np.hstack(images))
        # Retain every frame of the full lift/hold/lower/release candidate window.
        if f>=120:
            j=(f-120)%80
            if j==0:sheet=np.zeros((10*144,8*384,3),np.uint8);start=f
            tile=np.hstack([cv2.resize(img,(192,144)) for img in images])
            cv2.putText(tile,f'ep{ep} f{f} {f/30:.2f}s',(2,14),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
            y,x=divmod(j,8);sheet[y*144:(y+1)*144,x*384:(x+1)*384]=tile
            if j==79 or f==int(row.length)-1:
                path=epout/f'frames_{start:03d}_{f:03d}.jpg';cv2.imwrite(str(path),sheet)
                pages.append({'from':start,'to':f,'path':str(path)})
        decoded+=1
    writer.release()
    for cap in caps:cap.release()
    record={'episode':ep,'split':e['split'],'decoded_frames_per_camera':decoded,
                     'offset_frame':offsets,'clip':str(epout/'dual_view.mp4'),'pages':pages,
                     'review_status':'DECODED_NOT_REVIEWED'}
    print(e['split'],ep,decoded,flush=True)
    return record
with ThreadPoolExecutor(max_workers=4) as pool:
    for group in ['dev','train']:
        manifest.extend(pool.map(decode_episode,[e for e in allowed if e['split']==group]))
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
