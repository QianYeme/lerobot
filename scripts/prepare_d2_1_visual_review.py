"""Extract timestamped dual-view sheets for permitted D2.1 episodes only."""
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parents[1]
dataset=root/'数据集/formal4/kind_merged_nomaster'
out=root/'results/experiments/D2_1_ACTION_CHAIN_EVIDENCE_COMPLETION/visual_review'
out.mkdir(parents=True,exist_ok=True)
split=json.loads((root/'experiments/data/d1_training_readiness/index1/result/split_manifest_v1.json').read_text())
episodes=sorted([e for e in split['episodes'] if e['split'] in ('train','dev')],key=lambda e:(e['split']!='dev',e['episode_index']))
meta=pd.read_parquet(dataset/'meta/episodes/chunk-000/file-000.parquet',filters=[('episode_index','in',[e['episode_index'] for e in episodes])])
manifest=[]
for e in episodes:
    ep=e['episode_index'];row=meta[meta.episode_index==ep].iloc[0]
    frames=list(range(0,int(row.length),30))
    sheet=np.zeros((len(frames)*150,400,3),dtype=np.uint8)
    ok=0
    for ci,key in enumerate(['observation.images.top','observation.images.gripper']):
        p=dataset/'videos'/key/f"chunk-{int(row['videos/'+key+'/chunk_index']):03d}"/f"file-{int(row['videos/'+key+'/file_index']):03d}.mp4"
        cap=cv2.VideoCapture(str(p))
        offset=float(row['videos/'+key+'/from_timestamp'])
        for i,f in enumerate(frames):
            cap.set(cv2.CAP_PROP_POS_MSEC,(offset+f/30)*1000)
            success,img=cap.read()
            if success:
                img=cv2.resize(img,(200,150));ok+=1
                cv2.putText(img,f'ep{ep} f{f} {key.split(".")[-1]}',(3,14),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,255),1)
                sheet[i*150:(i+1)*150,ci*200:(ci+1)*200]=img
        cap.release()
    cv2.imwrite(str(out/f'ep{ep:03d}.jpg'),sheet)
    manifest.append({'split':e['split'],'episode':ep,'sample_frames':frames,'decoded_images':ok,
                     'expected_images':len(frames)*2,'review_status':'EXTRACTED_NOT_YET_REVIEWED',
                     'sheet':str(out/f'ep{ep:03d}.jpg'),'sampling_interval_s':1})
    print(ep,ok,flush=True)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
