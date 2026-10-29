"""Validate deployment wrappers on frozen Dev15, without hardware or Final15."""
import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import p4_1_pilot as old
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.p43_runtime import P43Runtime, FIELDS
from p43_export_robot_bundle import ROOT, OUT, SOURCES


def write_csv(name, rows):
    with (OUT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    split=ROOT/'experiments/p4_final_p1_training/p4_0_1_training_readiness_unblock/index1/result/dev15_episode_ids.json'
    ids=json.loads(split.read_text())
    if isinstance(ids,dict):
        ids=ids.get('episode_ids',ids.get('ids'))
    assert len(ids)==15
    data=old.Mapped(ids)
    meta=pd.read_parquet(old.DATA/'meta/episodes/chunk-000/file-000.parquet',filters=[('episode_index','in',ids)])
    assert set(meta.episode_index)==set(ids)
    action_std=torch.load(OUT/'assets/train90_statistics.pt',weights_only=False)['action']['std'].cuda()
    metrics=[];invariance=[];latency=[]
    for model in SOURCES:
        runtime=P43Runtime(OUT,model)
        if model=='F3':
            path=ROOT/'experiments/p4_final_p1_training/p4_2_1_dino_repair_parallel/index1/code/src/lerobot/policies/act/modeling_act.py'
            spec=importlib.util.spec_from_file_location('p43_fixed_reference',path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            cls=module.ACTPolicy
        else:
            cls=ACTPolicy
        reference=cls(runtime.policy.config).cuda().eval().requires_grad_(False)
        source=torch.load(SOURCES[model],map_location='cpu',weights_only=False)
        reference.load_state_dict(source['model'],strict=True);del source
        offset=0;max_diff=0.;times=[]
        for episode in ids:
            length=int(meta[meta.episode_index==episode].iloc[0].length)
            runtime.reset();reference.reset()
            for fraction in (0,.15,.3,.5,.7,.9):
                raw=data[offset+min(length-1,int(fraction*length))]
                obs=dict(zip(FIELDS,raw['observation.state'].tolist()))
                for camera in ('top','gripper'):
                    obs[camera]=(raw['observation.images.'+camera].permute(1,2,0).numpy()*255).round().astype(np.uint8)
                with torch.inference_mode():
                    batch=runtime.pre(raw)
                    if model=='F3':
                        batch['observation.images.top_raw']=raw['observation.images.top'][None].cuda()
                    target=reference.predict_action_chunk(batch)
                    torch.cuda.synchronize();start=time.perf_counter()
                    pred=runtime.chunk(obs)
                    torch.cuda.synchronize();times.append(time.perf_counter()-start)
                    diff=float((target-pred).abs().max());max_diff=max(max_diff,diff)
                    valid=~batch['action_is_pad'][0]
                    error=(pred[0]-batch['action'][0]).abs()[valid]
                    physical=error*action_std
                    metrics.append(dict(model=model,episode=episode,progress_proxy=fraction,
                        arm_normalized=float(error[:,:5].mean()),gripper_normalized=float(error[:,5].mean()),
                        arm_raw=float(physical[:,:5].mean()),gripper_raw=float(physical[:,5].mean()),max_reference_difference=diff))
            offset+=length
        assert max_diff<=1e-5,(model,max_diff)
        runtime.reset()
        first=runtime.step(obs)[0];runtime.reset();again=runtime.step(obs)[0]
        reset_diff=float(np.abs(first-again).max());assert reset_diff<=1e-5
        invariance.append(dict(model=model,dev_samples=90,max_abs_normalized=max_diff,tolerance=1e-5,
                               reset_max_abs_native=reset_diff,status='PASS'))
        latency.append(dict(model=model,samples=len(times),mean_ms=float(np.mean(times)*1000),
                            p95_ms=float(np.percentile(times,95)*1000),scope='GPU wrapper chunk only; no camera/serial'))
        del reference,runtime;torch.cuda.empty_cache()
        print(model,'PASS',max_diff,flush=True)
    write_csv('deployed_runtime_dev15.csv',metrics)
    write_csv('runtime_invariance.csv',invariance)
    write_csv('runtime_latency.csv',latency)
    (OUT/'deployment_preflight.json').write_text(json.dumps(dict(status='OFFLINE_PASS_HARDWARE_PENDING',
        dev_episodes=ids,models=list(SOURCES),final15_accessed=False,robot_operated=False,
        scope='90 fixed progress samples per model; teacher-forced observations, not robot success'),indent=2)+'\n')


if __name__=='__main__':
    main()
