"""Isolated DINO-input repair gates, disposable benchmark, and fresh fixed workers."""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

import p4_1_pilot as old
import p4_2_queue as queue

ROOT = old.ROOT
WORK = ROOT / 'experiments/p4_final_p1_training/p4_2_1_dino_repair_parallel/index1'
OUT = WORK / 'result'
P41 = queue.P41
STANDARD = old.make_act_pre_post_processors


class RawTopProcessor:
    def __init__(self, standard, device, legacy=False):
        self.standard = standard
        self.device = device
        self.legacy = legacy

    def __call__(self, batch):
        raw = batch['observation.images.top'].clone()
        processed = self.standard(batch)
        if self.legacy:
            processed['observation.images.top_raw'] = processed['observation.images.top']
        else:
            if raw.ndim == 3:
                raw = raw.unsqueeze(0)
            processed['observation.images.top_raw'] = raw.to(self.device)
        return processed


def processors(config, stats):
    pre, post = STANDARD(config, stats)
    return RawTopProcessor(pre, config.device), post


def assets(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name in ('formal6_train90_prototypes.pt', 'prototype_provenance.json', 'train90_statistics.pt'):
        shutil.copyfile(P41/name, folder/name)
    old.OUT = folder
    cfg = old.config()
    stats = torch.load(P41/'train90_statistics.pt', map_location='cpu', weights_only=False)
    return cfg, stats


def preflight():
    assert '/p4_2_1_dino_repair_parallel/' in __import__('lerobot').__file__, 'ISOLATION_FAILED'
    folder = OUT/'preflight'
    cfg, stats = assets(folder)
    queue.reseed(1)
    model = old.ACTPolicy(cfg).cuda()
    initial_hash = queue.parameter_hash(model)
    (folder/'initial_model_hash.json').write_text(json.dumps({'seed':1,'hash':initial_hash}))
    pre, post = processors(cfg, stats)
    baseline, _ = STANDARD(cfg, stats)
    encoder = model.model.object_conditioning.dino
    original_forward = encoder.model.forward_features
    captured = {}
    def observe(image, *args, **kwargs):
        result = original_forward(image, *args, **kwargs)
        captured['input'] = image.detach().clone()
        captured['patches'] = result['x_norm_patchtokens'].detach().clone()
        return result
    encoder.model.forward_features = observe
    cnn = []
    handle = model.model.backbone.register_forward_pre_hook(lambda _m, args: cnn.append(args[0].detach().clone()))
    try:
        for ep in (0,29,58):
            raw = old.Mapped([ep])[0]
            reference = baseline(dict(raw))
            fixed = pre(dict(raw))
            for camera in ('top','gripper'):
                key='observation.images.'+camera
                diff=float((fixed[key]-reference[key]).abs().max())
                assert diff<=1e-6
                queue.OUT=OUT
                queue.append('resnet_input_invariance.csv',dict(episode=ep,frame=0,camera=camera,max_abs_diff=diff))
            cnn.clear()
            with torch.inference_mode():
                prediction = model.predict_action_chunk(fixed)
                policy_input=captured['input'].clone()
                policy_patches=captured['patches'].clone()
                encoder(raw['observation.images.top'][None].cuda())
            assert prediction.shape==(1,100,6)
            diff=float((policy_input-captured['input']).abs().max())
            feature_diff=float((policy_patches-captured['patches']).abs().max())
            assert diff<=1e-5 and feature_diff<=1e-5
            assert len(cnn)==2
            for index,key in enumerate(cfg.image_features):
                torch.testing.assert_close(cnn[index],reference[key],atol=1e-6,rtol=0)
            queue.append('preprocess_equivalence.csv',dict(episode=ep,frame=0,shape=str(list(policy_input.shape)),
                dtype=str(policy_input.dtype),minimum=float(policy_input.min()),maximum=float(policy_input.max()),
                policy_prototype_input_max_diff=diff,feature_max_diff=feature_diff,output_shape=str(list(prediction.shape))))
    finally:
        handle.remove()
        encoder.model.forward_features=original_forward
    assert not any(p.requires_grad for p in encoder.parameters())
    ids=json.loads((old.PRIOR/'train90_episode_ids.json').read_text())
    dataset=old.Mapped(ids)
    optimizer=torch.optim.AdamW(model.get_optim_params(),lr=1e-5,weight_decay=1e-4)
    start=time.monotonic()
    for step,raw in enumerate(queue.batches(dataset,1,0,0),1):
        batch=pre(raw)
        assert batch['observation.state'].shape==(4,8) and batch['action'].shape==(4,100,6)
        loss,grad=queue.update(model,optimizer,batch)
        queue.append('smoke_metrics.csv',dict(step=step,loss=loss,grad=grad,elapsed_s=time.monotonic()-start))
        if step==100:
            break
    path=old.checkpoint(model,optimizer,100,0,100,cfg,'smoke')
    with torch.inference_mode():
        expected=model.predict_action_chunk(batch).cpu()
    loaded=torch.load(path,map_location='cpu',weights_only=False)
    reloaded=old.ACTPolicy(cfg).cuda()
    reloaded.load_state_dict(loaded['model'])
    opt2=torch.optim.AdamW(reloaded.get_optim_params(),lr=1e-5,weight_decay=1e-4)
    opt2.load_state_dict(loaded['optimizer'])
    with torch.inference_mode():
        actual=reloaded.predict_action_chunk(batch).cpu()
    torch.testing.assert_close(expected,actual,atol=0,rtol=0)
    queue.write('engineering_gate.json',dict(status='PASS',samples=[0,29,58],smoke_steps=100,
        restore_max_diff=float((expected-actual).abs().max()),source_isolation=True,raw_top_explicit=True,
        inverse_normalization=False,clip_applied=False,prototypes_sha=old.sha(P41/'formal6_train90_prototypes.pt'),
        normalization_sha=old.sha(P41/'train90_statistics.pt'),initial_model_hash=initial_hash))


def benchmark(seed, name, steps):
    cfg, stats=assets(OUT/'benchmarks'/name)
    queue.reseed(seed)
    model=old.ACTPolicy(cfg).cuda()
    optimizer=torch.optim.AdamW(model.get_optim_params(),lr=1e-5,weight_decay=1e-4)
    pre,_=processors(cfg,stats)
    ids=json.loads((old.PRIOR/'train90_episode_ids.json').read_text())
    loader=queue.batches(old.Mapped(ids),seed,0,0)
    start=None
    for step,raw in enumerate(loader,1):
        batch=pre(raw)
        queue.update(model,optimizer,batch)
        torch.cuda.synchronize()
        if step==100:
            start=time.monotonic()
            torch.cuda.reset_peak_memory_stats()
        if step==steps:
            seconds=time.monotonic()-start
            (OUT/'benchmarks'/name/'result.json').write_text(json.dumps(dict(seed=seed,steps=steps-100,
                measured_s=seconds,steps_per_s=(steps-100)/seconds,peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved(),pid=os.getpid(),warmup_excluded=100)))
            break


def train(seed, name, cutoff):
    assert json.loads((OUT/'engineering_gate.json').read_text())['status']=='PASS'
    queue.OUT=OUT/'workers'/name
    queue.OUT.mkdir(parents=True,exist_ok=True)
    old.make_act_pre_post_processors=processors
    ids=json.loads((old.PRIOR/'train90_episode_ids.json').read_text())
    dev=json.loads((old.PRIOR/'dev15_episode_ids.json').read_text())
    queue.run(name,seed,True,False,ids,dev,cutoff)


def evaluate_existing():
    queue.OUT=OUT
    # Wait only within the reserved window for read-only matched legacy reuse.
    schedule=json.loads((OUT/'controller_started.json').read_text())
    until=min(schedule['training_cutoff_monotonic'],schedule['finish_deadline_monotonic']-75*60)
    while time.monotonic()<until:
        pids=[31212]
        retry=ROOT/'experiments/p4_final_p1_training/p4_2_unattended_multi_model/index1/result/retry1/controller_identity.json'
        if retry.exists():pids.append(json.loads(retry.read_text())['pid'])
        alive=False
        for pid in pids:
            try:os.kill(pid,0);alive=True
            except ProcessLookupError:pass
        if not alive:break
        time.sleep(15)
    ids=json.loads((old.PRIOR/'dev15_episode_ids.json').read_text())
    p42=ROOT/'experiments/p4_final_p1_training/p4_2_unattended_multi_model/index1/result'
    candidates=[('L1',P41/'checkpoints/pilot/080000/resume.pt',True,True)]
    for label,run,p1 in [('L2','P2_P1_S2_80k',True),('L3','P3_P1_S3_80k',True),('ACT1','P1_ACT_S1_80k',False)]:
        path=p42/'runs'/run/'checkpoints/formal/080000/resume.pt'
        registry=p42/'retry_checkpoint_registry.json'
        if not path.exists() and registry.exists():
            candidate=json.loads(registry.read_text()).get(label)
            if candidate:path=Path(candidate)
        if path.exists():
            candidates.append((label,path,p1,True))
    for label,path,p1,legacy in candidates:
        cfg,stats=assets(OUT/'evaluation_assets'/label)
        if not p1:
            cfg.model_type='act';cfg.object_conditioning_enabled=False;cfg.object_conditioning_prototype_path=None
        model=old.ACTPolicy(cfg).cuda()
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model'])
        del saved
        standard,_=STANDARD(cfg,stats)
        pre=RawTopProcessor(standard,cfg.device,legacy=legacy)
        audited_evaluate(model,pre,stats,ids,label,80000,p1)
        queue.append('legacy_reuse_manifest.csv',dict(label=label,path=str(path),sha256=old.sha(path),step=80000))
        del model
        torch.cuda.empty_cache()
    # Fixed checkpoints are replayed through precisely the same scorer; keep raw worker results intact.
    for seed in (1,2,3):
        name='F'+str(seed)
        path=OUT/'workers'/name/'runs'/name/'checkpoints/formal/080000/resume.pt'
        if not path.exists():continue
        queue.OUT=OUT/'same_code_replay'/name
        queue.OUT.mkdir(parents=True,exist_ok=True)
        cfg,stats=assets(OUT/'evaluation_assets'/name)
        model=old.ACTPolicy(cfg).cuda()
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']);del saved
        pre,_=processors(cfg,stats)
        audited_evaluate(model,pre,stats,ids,name,80000,True)
        del model;torch.cuda.empty_cache()
    queue.OUT=OUT
    import pandas as pd
    paths=[OUT/'dev15_model_comparison.csv']+list((OUT/'same_code_replay').glob('*/dev15_model_comparison.csv'))
    frames=[pd.read_csv(p) for p in paths if p.exists()]
    if frames:
        frame=pd.concat(frames,ignore_index=True)
        assert not frame.duplicated(['run','step','episode','progress_proxy']).any()
        frame.to_csv(OUT/'dev15_same_code_audited_comparison.csv',index=False)
        frame.groupby(['run','step','episode'])[['arm_normalized','gripper_normalized']].agg(['mean','median']).to_csv(OUT/'same_code_per_episode.csv')
        means=frame.groupby('run')[['arm_normalized','gripper_normalized']].mean()
        means[means.index.isin(['F1','F2','F3'])].agg(['mean','std','count']).to_csv(OUT/'fixed_seed_variance.csv')


def audited_evaluate(model,pre,stats,ids,name,step,p1):
    if not p1:
        return queue.evaluate(model,pre,stats,ids,name,step,p1)
    original=model.predict_action_chunk
    context={'calls':0,'on':None}
    def observe(batch):
        result=original(batch)
        index=context['calls']//2
        if context['calls']%2==0:
            context['on']=result.detach().clone()
        else:
            delta=context['on']-result
            ep=ids[index//6]
            fraction=(0,.15,.3,.5,.7,.9)[index%6]
            values=dict(run=name,step=step,episode=ep,progress_proxy=fraction,
                h0_pan_signed_on_off=float(delta[0,0,0]),h0_gripper_signed_on_off=float(delta[0,0,5]),
                h0_pan_raw_signed_on_off=float(delta[0,0,0]*stats['action']['std'][0].cuda()),
                h0_gripper_raw_signed_on_off=float(delta[0,0,5]*stats['action']['std'][5].cuda()),
                chunk_mean_abs_on_off=float(delta.abs().mean()))
            path=OUT/'object_path_signed_diagnostics.csv'
            present=path.exists()
            with path.open('a',newline='') as stream:
                import csv
                writer=csv.DictWriter(stream,fieldnames=list(values))
                if not present:writer.writeheader()
                writer.writerow(values)
        context['calls']+=1
        return result
    model.predict_action_chunk=observe
    try:
        return queue.evaluate(model,pre,stats,ids,name,step,p1)
    finally:
        model.predict_action_chunk=original


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['preflight','benchmark','train','evaluate'])
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--name',default='single')
    parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--cutoff',type=float,default=0)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    queue.OUT=OUT
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(False)
    if args.mode=='preflight':preflight()
    elif args.mode=='benchmark':benchmark(args.seed,args.name,args.steps)
    elif args.mode=='train':train(args.seed,args.name,args.cutoff)
    else:evaluate_existing()


if __name__=='__main__':
    main()
