"""Native official Diffusion: temporal adapter, isolated smoke and conditional full budget."""
import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
import random
import signal
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors

ROOT=Path('/root/autodl-tmp/lerobot-main')
PARENT=ROOT/'experiments/p4_final_p1_training/p4_5_robot_reliability_server10h/index1'
WORK=PARENT.parent/'supplements/diffusion_baseline/index1'
OUT=WORK/'result';DATA=ROOT/'outputs/p4_0/formal6_merged'
PRIOR=ROOT/'experiments/p4_final_p1_training/p4_0_1_training_readiness_unblock/index1/result'
FIELDS=['shoulder_pan.pos','shoulder_lift.pos','elbow_flex.pos','wrist_flex.pos','wrist_roll.pos','gripper.pos']
STOP=False


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def row(path,value):
    exists=path.exists()
    with path.open('a',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(value))
        if not exists:writer.writeheader()
        writer.writerow(value)


def config(device):
    assert '/stock/src/' in inspect.getfile(DiffusionPolicy)
    return DiffusionConfig(device=device,input_features={
        'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(6,)),
        **{'observation.images.'+name:PolicyFeature(type=FeatureType.VISUAL,shape=(3,480,640)) for name in ['gripper','top']}},
        output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(6,))})


class Mapped(Dataset):
    def __init__(self,ids,cfg):
        times=[i/30 for i in cfg.observation_delta_indices]
        deltas={key:times for key in cfg.input_features}
        deltas['action']=[i/30 for i in cfg.action_delta_indices]
        self.base=LeRobotDataset('QYyyyyyyy/formal6_merged',root=DATA,episodes=ids,delta_timestamps=deltas,video_backend='torchcodec')
        self.indices=[self.base.meta.features['observation.state']['names'].index(name) for name in FIELDS]
        self.meta=pd.read_parquet(DATA/'meta/episodes/chunk-000/file-000.parquet',filters=[('episode_index','in',ids)])
        self.lengths=[int(self.meta[self.meta.episode_index==ep].iloc[0].length) for ep in ids]
        assert sum(self.lengths)==len(self.base)

    def __len__(self):return len(self.base)

    def __getitem__(self,index):
        raw=self.base[index];raw['observation.state']=raw['observation.state'][...,self.indices]
        return {key:raw[key] for key in raw if key in ['observation.state','observation.images.top','observation.images.gripper','action','episode_index','frame_index'] or key.endswith('_is_pad')}


def statistics(ids):
    frame=pd.read_parquet(DATA/'data/chunk-000/file-000.parquet',columns=['observation.state','action','episode_index'],filters=[('episode_index','in',ids)])
    names=json.loads((DATA/'meta/info.json').read_text())['features']['observation.state']['names']
    stats={}
    for key in ['observation.state','action']:
        values=np.stack(frame[key])
        if key=='observation.state':values=values[:,[names.index(name) for name in FIELDS]]
        assert values.shape[1]==6 and np.isfinite(values).all()
        stats[key]={name:torch.tensor(value) for name,value in [('min',values.min(0)),('max',values.max(0)),('mean',values.mean(0)),('std',values.std(0).clip(1e-6))]}
    for camera in ['top','gripper']:
        stats['observation.images.'+camera]=dict(mean=torch.tensor([.485,.456,.406]).reshape(3,1,1),std=torch.tensor([.229,.224,.225]).reshape(3,1,1))
    return stats


def rng():
    return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],python=random.getstate(),numpy=np.random.get_state())


def restore(state):
    torch.set_rng_state(state['cpu']);random.setstate(state['python']);np.random.set_state(state['numpy'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])


def update(policy,opt,scheduler,pre,raw,clip):
    policy.train();opt.zero_grad(set_to_none=True);loss,_=policy(pre(raw));assert torch.isfinite(loss)
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),clip,error_if_nonfinite=True)
    opt.step();scheduler.step();return float(loss.detach()),float(norm)


@torch.inference_mode()
def predict(policy,batch):
    # Native generate_actions receives the dataset's actual two-frame history.
    inputs={'observation.state':batch['observation.state'],'observation.images':torch.stack([batch[key] for key in policy.config.image_features],dim=-4)}
    return policy.diffusion.generate_actions(inputs)


def smoke(folder,cfg,policy,opt,scheduler,pre,stats,dataset):
    raw=next(iter(DataLoader(dataset,batch_size=1,num_workers=0)))
    assert raw['observation.state'].shape==(1,cfg.n_obs_steps,6)
    assert raw['action'].shape==(1,cfg.horizon,6)
    assert torch.equal(raw['observation.state'][0,0],raw['observation.state'][0,1]),'First-frame past padding must replicate episode start'
    tail=dataset[dataset.lengths[0]-1]
    assert int(tail['action_is_pad'].sum())>0
    batch=pre(raw)
    expected=(raw['observation.state'].to(cfg.device)-stats['observation.state']['min'].to(cfg.device))/(stats['observation.state']['max'].to(cfg.device)-stats['observation.state']['min'].to(cfg.device)).clamp_min(1e-8)*2-1
    assert torch.allclose(batch['observation.state'],expected,atol=1e-5)
    loss,grad=update(policy,opt,scheduler,pre,raw,cfg.get_optimizer_preset().grad_clip_norm)
    path=folder/'smoke_resume.pt'
    torch.save(dict(model=policy.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=rng()),path)
    expected_loss,_=update(policy,opt,scheduler,pre,raw,cfg.get_optimizer_preset().grad_clip_norm)
    h=hashlib.sha256()
    for key,value in policy.state_dict().items():h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    target_hash=h.hexdigest()
    saved=torch.load(path,map_location='cpu',weights_only=False);policy.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);restore(saved['rng']);del saved
    actual_loss,_=update(policy,opt,scheduler,pre,raw,cfg.get_optimizer_preset().grad_clip_norm)
    h=hashlib.sha256()
    for key,value in policy.state_dict().items():h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    assert target_hash==h.hexdigest(),(expected_loss,actual_loss)
    policy.eval();torch.manual_seed(20261030);began=time.perf_counter();prediction=predict(policy,batch);infer_s=time.perf_counter()-began
    current={key:batch[key][:,-1] for key in cfg.input_features}
    policy.reset();torch.manual_seed(20261030)
    deployed=policy.select_action(current)
    difference=float((deployed-prediction[:,0]).abs().max());assert difference<=1e-5
    dev=Mapped(json.loads((PRIOR/'dev15_episode_ids.json').read_text()),cfg)
    dev_raw=next(iter(DataLoader(dev,batch_size=1,num_workers=0)));torch.manual_seed(20261030)
    with torch.inference_mode():dev_pred=predict(policy,pre(dev_raw))
    assert torch.isfinite(dev_pred).all()
    write(folder/'smoke.json',dict(status='PASS',device=cfg.device,state_shape=list(raw['observation.state'].shape),action_shape=list(raw['action'].shape),
        first_frame_padding=True,end_action_pad_count=int(tail['action_is_pad'].sum()),optimizer_scheduler_rng_next_update_bitwise=True,
        first_action_live_contract_max_difference=difference,inference_s=infer_s,dev_forward=True,loss=loss,grad_norm=grad,
        parameters=sum(p.numel() for p in policy.parameters()),native_denoise_steps=policy.diffusion.num_inference_steps,
        no_final15=True,disposable_updates=True))


def make_loader(dataset,cfg,cursor=0):
    ends=np.cumsum(dataset.lengths).tolist();starts=[0]+ends[:-1]
    sampler=EpisodeAwareSampler(starts,ends,drop_n_last_frames=cfg.drop_n_last_frames,shuffle=True)
    from lerobot.datasets import video_utils
    def clear(_):video_utils._default_decoder_cache.clear()
    dl=DataLoader(dataset,batch_size=8,num_workers=4,pin_memory=True,sampler=sampler,drop_last=False,worker_init_fn=clear)
    iterator=iter(dl)
    for _ in range(cursor):next(iterator)
    return iterator


@torch.inference_mode()
def evaluate(policy,pre,post,run):
    cfg=policy.config;policy.eval();dev=Mapped(json.loads((PRIOR/'dev15_episode_ids.json').read_text()),cfg);offset=0
    for ep,length in zip(dev.base.episodes,dev.lengths):
        for fraction in [0,.15,.3,.5,.7,.9]:
            index=offset+min(length-1,int(fraction*length));raw=next(iter(DataLoader(torch.utils.data.Subset(dev,[index]),batch_size=1,num_workers=0)))
            torch.manual_seed(20261030+int(ep)*1000+int(fraction*100))
            batch=pre(raw);start=time.perf_counter();pred=predict(policy,batch);native=post(pred).cpu()
            # Native predictions start at t, while dataset target starts at t-1.
            target=raw['action'][:,cfg.n_obs_steps-1:cfg.n_obs_steps-1+cfg.n_action_steps]
            error=(native-target).abs();valid=~raw['action_is_pad'][:,cfg.n_obs_steps-1:cfg.n_obs_steps-1+cfg.n_action_steps]
            row(OUT/'diffusion_dev15.csv',dict(run=run,episode=int(ep),progress_proxy=fraction,arm_h0_raw=float(error[0,0,:5].mean()),
                gripper_h0_raw=float(error[0,0,5]),arm_execution_window_raw=float(error[valid][:,:5].mean()),
                gripper_execution_window_raw=float(error[valid][:,5].mean()),inference_s=time.perf_counter()-start,noise_seed=20261030+int(ep)*1000+int(fraction*100)))
            row(OUT/'baseline_h0_comparison.csv',dict(run=run,episode=int(ep),progress_proxy=fraction,
                arm_h0_raw=float(error[0,0,:5].mean()),gripper_h0_raw=float(error[0,0,5])))
        offset+=length


@torch.inference_mode()
def compare_act():
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    dc=config('cuda');dev=Mapped(json.loads((PRIOR/'dev15_episode_ids.json').read_text()),dc)
    for name in ['VACT-S1','VACT-S2','VACT-S3']:
        folder=PARENT/'result/server/runs'/name
        saved=torch.load(folder/'checkpoints/080000/resume.pt',map_location='cpu',weights_only=False)
        actor=ACTPolicy(saved['config']).cuda().eval();actor.load_state_dict(saved['model']);del saved
        stats=torch.load(folder/'statistics.pt',weights_only=False);pre,post=make_act_pre_post_processors(actor.config,stats)
        offset=0
        for ep,length in zip(dev.base.episodes,dev.lengths):
            for fraction in [0,.15,.3,.5,.7,.9]:
                raw=next(iter(DataLoader(torch.utils.data.Subset(dev,[offset+min(length-1,int(fraction*length))]),batch_size=1,num_workers=0)))
                current={key:raw[key][:,-1] for key in actor.config.input_features}
                pred=post(actor.predict_action_chunk(pre(current)))[:,0]
                error=(pred-raw['action'][:,dc.n_obs_steps-1]).abs()
                row(OUT/'baseline_h0_comparison.csv',dict(run=name,episode=int(ep),progress_proxy=fraction,
                    arm_h0_raw=float(error[0,:5].mean()),gripper_h0_raw=float(error[0,5])))
            offset+=length
        del actor;torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--mode',choices=['cpu-smoke','gpu-benchmark','train'],required=True)
    p.add_argument('--seed',type=int,default=1);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    signal.signal(signal.SIGTERM,lambda *_:globals().__setitem__('STOP',True))
    torch.set_num_threads(1 if a.mode=='cpu-smoke' else 4)
    if a.mode!='cpu-smoke':
        torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    cfg=config('cpu' if a.mode=='cpu-smoke' else 'cuda');ids=json.loads((PRIOR/'train90_episode_ids.json').read_text())
    folder=OUT/a.run;folder.mkdir(parents=True,exist_ok=True);cfg.save_pretrained(folder/'config')
    stats=statistics(ids);torch.save(stats,folder/'statistics.pt');pre,post=make_diffusion_pre_post_processors(cfg,stats)
    policy=DiffusionPolicy(cfg).to(cfg.device);preset=cfg.get_optimizer_preset();opt=preset.build(policy.get_optim_params())
    budget=TrainPipelineConfig.__dataclass_fields__['steps'].default
    scheduler=cfg.get_scheduler_preset().build(opt,budget);dataset=Mapped(ids,cfg)
    write(folder/'identity.json',dict(source=inspect.getfile(DiffusionPolicy),official_commit='8fff0fde7c79f23a93d845d1a50e985de01f8b8a',
        seed=a.seed,native_default_budget=budget,target_steps=budget,batch_size=8,observation_fields=FIELDS,
        config=json.loads((folder/'config/config.json').read_text()),normalizer_source='Train90 only; state/action MIN_MAX; image ImageNet',
        no_project_additions=True,mode=a.mode,formal=a.mode=='train'))
    if a.mode=='cpu-smoke':smoke(folder,cfg,policy,opt,scheduler,pre,stats,dataset);return
    step=epoch=cursor=0;samples=0
    if (folder/'latest.json').exists() and a.mode=='train':
        saved=torch.load(json.loads((folder/'latest.json').read_text())['checkpoint'],map_location='cpu',weights_only=False)
        assert saved['split_sha256']==hashlib.sha256((PRIOR/'split_freeze_manifest.json').read_bytes()).hexdigest()
        policy.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        step,epoch,cursor,samples=saved['step'],saved['epoch'],saved['cursor'],saved['samples']
        torch.set_rng_state(saved['epoch_cpu_rng']);iterator=make_loader(dataset,cfg,cursor);epoch_rng=saved['epoch_cpu_rng'];restore(saved['rng'])
    else:epoch_rng=torch.get_rng_state();iterator=make_loader(dataset,cfg)
    start=time.perf_counter();target=50 if a.mode=='gpu-benchmark' else budget
    def save():
        path=folder/'checkpoints'/f'{step:06d}'/'resume.pt';path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
        torch.save(dict(model=policy.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=rng(),
            step=step,epoch=epoch,cursor=cursor,epoch_cpu_rng=epoch_rng,samples=samples,config=cfg,
            split_sha256=hashlib.sha256((PRIOR/'split_freeze_manifest.json').read_bytes()).hexdigest(),
            dataset_revision='83a829af6df245ed53596dbc22b18c19f8bd3315'),tmp);tmp.replace(path)
        write(folder/'latest.json',dict(checkpoint=str(path),step=step))
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        row(OUT/'checkpoint_manifest.csv',dict(run=a.run,step=step,path=str(path),bytes=path.stat().st_size,sha256=digest,optimizer_rng_scheduler=True))
    while step<target and time.time()<a.deadline and not STOP:
        try:raw=next(iterator)
        except StopIteration:epoch+=1;cursor=0;epoch_rng=torch.get_rng_state();iterator=make_loader(dataset,cfg);raw=next(iterator)
        loss,grad=update(policy,opt,scheduler,pre,raw,preset.grad_clip_norm);torch.cuda.synchronize();step+=1;cursor+=1;samples+=len(raw['action'])
        if step==1 or step%100==0:row(OUT/('training_metrics.csv' if a.mode=='train' else 'gpu_benchmark_metrics.csv'),dict(run=a.run,step=step,loss=loss,grad_norm=grad,samples=samples,elapsed_s=time.perf_counter()-start,lr=opt.param_groups[0]['lr']))
        if a.mode=='train' and step%20000==0:save()
    elapsed=time.perf_counter()-start
    if a.mode=='gpu-benchmark':
        single={k:v[:1] for k,v in raw.items()};policy.eval();torch.cuda.synchronize();began=time.perf_counter()
        with torch.inference_mode():prediction=predict(policy,pre(single))
        torch.cuda.synchronize();inference_s=time.perf_counter()-began;assert torch.isfinite(prediction).all()
        write(folder/'benchmark.json',dict(steps=step,elapsed_s=elapsed,steps_per_s=step/elapsed,
            native_inference_s=inference_s,peak_vram=torch.cuda.max_memory_allocated()));return
    if not (folder/'latest.json').exists() or json.loads((folder/'latest.json').read_text())['step']!=step:save()
    if step==budget:
        policy.save_pretrained(folder/'pretrained_model');evaluate(policy,pre,post,a.run)
        if a.seed==1:compare_act()
    write(folder/'execution_complete.json',dict(status='DIFFUSION_COMPLETE' if step==budget else 'DIFFUSION_PARTIAL',steps=step,
        native_default_budget=budget,walltime_s=elapsed,gpu_hours=elapsed/3600,samples=samples,no_final15=True))


if __name__=='__main__':main()
