"""Identical within-pair Dev scoring and frozen native LIBERO init lists."""
import argparse
import json
import os
import shutil
import time

from p49_runtime import (OUT, JOBS, R5, baseline, backend, factory, row, sha, write)

def load(job, historical):
    import torch
    folder=OUT/'evaluation_assets'/job/('gate0' if historical else 'gateplus')
    old,cfg,policy,opt,data,pre,stats=factory(job,folder)
    del opt,data
    if historical:
        path=baseline(job)
    else:
        latest=json.loads((OUT/'runs'/job/'latest.json').read_text())
        assert latest['step']==JOBS[job][3]
        path=latest['path'];assert sha(path)==latest['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['step']==JOBS[job][3]
    policy.load_state_dict(saved['model'],strict=True);del saved;policy.eval()
    return old,cfg,policy,pre,stats,path

def dev(job,historical,act=False):
    import numpy as np
    import torch
    if act:
        old=backend(job);cfg,policy,receipt=stock_load();pre=None
        stats=old.CachedData('dev').stats;path=receipt['path']
    else:
        old,cfg,policy,pre,stats,path=load(job,historical)
    domain,kind,seed,target=JOBS[job]
    label='ACT_GATE_NA' if act else job+('_GATE0' if historical else '')
    if domain=='lb':
        data=old.CachedData('dev')
        observations=[(int(ep),int(data.arrays['frame'][i]),[i])
            for i,ep in enumerate(data.arrays['episode'])]
    else:
        import pandas as pd
        ids=json.loads((old.PRIOR/'dev15_episode_ids.json').read_text())
        data=old.Mapped(ids);meta=pd.read_parquet(old.DATA/'meta/episodes/chunk-000/file-000.parquet')
        observations=[];offset=0
        for ep in ids:
            length=int(meta[meta.episode_index==ep].iloc[0].length)
            for fraction in (0,.15,.3,.5,.7,.9):
                frame=min(length-1,int(fraction*length));observations.append((ep,frame,offset+frame))
            offset+=length
    std=stats['action']['std'].cuda();arm=6 if domain=='lb' else 5
    folder=OUT/'evaluation'/label;folder.mkdir(parents=True,exist_ok=True)
    trace={}
    oc=None if act else policy.model.object_conditioning
    def capture(_m,args,output):
        trace['query_norm']=float(args[0].norm());trace['raw_residual_norm']=float(output[0].norm())
    def tokens(_m,args,output):
        trace['prototype_max_similarity']=float(output[1].max())
        trace['token_shape']=list(output[0].shape)
        trace['_similarity']=output[1].detach().cpu().numpy()
    handles=[] if act else [oc.cross_attention.register_forward_hook(capture),oc.tokenizer.register_forward_hook(tokens)]
    evidence=[];predictions=[]
    try:
        for ep,frame,index in observations:
            batch=data.batch(index) if domain=='lb' else pre(data[index])
            with torch.inference_mode():
                pred=policy.predict_action_chunk(batch);values={k:v for k,v in trace.items() if not k.startswith('_')}
                if not act:
                    similarity=trace['_similarity'].copy()
                    gate=oc.gate.detach().clone()
                    try:oc.gate.zero_();off=policy.predict_action_chunk(batch)
                    finally:oc.gate.copy_(gate)
            valid=~batch['action_is_pad'][0];err=(pred[0]-batch['action'][0]).abs()
            for scope,e in [('h0',err[:1]),('chunk_valid',err[valid])]:
                raw=e*std
                row(folder/'dev_metrics.csv',dict(model=label,episode=ep,frame=frame,scope=scope,
                    arm_normalized_mae=float(e[:,:arm].mean()),gripper_normalized_mae=float(e[:,arm].mean()),
                    arm_raw_mae=float(raw[:,:arm].mean()),gripper_raw_mae=float(raw[:,arm].mean()),
                    finite=bool(torch.isfinite(pred).all()),action_normalized_min=float(pred.min()),
                    action_normalized_max=float(pred.max())))
            predictions.append(dict(episode=ep,frame=frame,h0=pred[0,0].cpu().tolist()))
            if act:continue
            delta=(pred-off).abs()[0]
            record=dict(model=label,episode=ep,frame=frame,gate=float(gate),
                gated_residual_query_ratio=abs(float(gate))*values['raw_residual_norm']/max(values['query_norm'],1e-12),
                on_off_h0_arm_normalized=float(delta[0,:arm].mean()),
                on_off_h0_gripper_normalized=float(delta[0,arm]),
                on_off_h0_arm_raw=float((delta[0]*std)[:arm].mean()),
                on_off_h0_gripper_raw=float((delta[0]*std)[arm]),**values)
            evidence.append(record)
            if len(evidence) in [1,len(observations)//2,len(observations)]:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig,ax=plt.subplots(figsize=(4,4));ax.imshow(similarity[0]);ax.set_title(f'{label} ep{ep} f{frame} similarity')
                fig.tight_layout();fig.savefig(folder/f'prototype_response_ep{ep}_f{frame}.png');plt.close(fig)
        write(folder/'object_path.json',evidence);write(folder/'h0_predictions.json',predictions)
    finally:
        for h in handles:h.remove()
    write(folder/'dev_complete.json',dict(status='DONE',model=label,checkpoint=str(path),sha256=sha(path),
        observation_count=len(observations),sampling='all1391 Dev9 frames' if domain=='lb' else 'six fixed historical fractions per Dev15 episode; 90 observations',
        teacher_forced=True,real_robot_evidence=False))

def libero(job,historical,act=False):
    backend(job)
    import p48_r5_evaluate as evaluator
    import p48_r5_common as old
    label='ACT_GATE_NA' if act else job+('_GATE0' if historical else '')
    if act:
        load_fn=lambda _kind:stock_load()
    else:
        def load_fn(_kind):
            _,cfg,policy,_,_,path=load(job,historical)
            return cfg,policy,dict(path=str(path),sha256=sha(path),step=10000)
    evaluator.load=load_fn
    # Run untouched native evaluator twice with disjoint init lists, same physics and budget.
    for cohort in ['reused','new']:
        dest=OUT/'libero_rollout_jobs'/label/cohort;dest.mkdir(parents=True,exist_ok=True)
        contract=json.loads((R5/'result/timing_contract.json').read_text())
        frozen=json.loads((OUT/'libero_init_contract.json').read_text())
        contract['paired_initial_ids']=frozen[cohort]
        contract['episode_seeds']=[1]*len(frozen[cohort])
        write(dest/'timing_contract.json',contract)
        shutil.copyfile(R5/'result/train35_statistics.pt',dest/'train35_statistics.pt')
        evaluator.OUT=dest
        evaluator.rollout('act' if act else JOBS[job][1])
    write(OUT/'evaluation'/label/'rollout_complete.json',dict(status='DONE',model=label,cohorts=['reused','new']))

def stock_load():
    import torch
    import p48_r5_common as old
    cfg,policy=old.make_policy('act')
    path=R5/'result/runs/act/checkpoints/010000/resume.pt'
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['step']==10000
    policy.load_state_dict(saved['model'],strict=True);policy.eval()
    return cfg,policy,dict(path=str(path),sha256=sha(path),step=10000)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',choices=list(JOBS),required=True)
    p.add_argument('--historical',action='store_true');p.add_argument('--mode',choices=['dev','rollout','act_dev','act_rollout'],required=True)
    a=p.parse_args()
    if a.mode=='dev':dev(a.job,a.historical)
    elif a.mode=='rollout':libero(a.job,a.historical)
    elif a.mode=='act_rollout':libero(a.job,True,True)
    else:dev(a.job,True,act=True)
