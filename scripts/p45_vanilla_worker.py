"""P4.5 stock ACT runner. Execute with the pinned official stock/src on PYTHONPATH."""
import argparse
import csv
import hashlib
import inspect
import json
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets import video_utils
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors

ROOT=Path('/root/autodl-tmp/lerobot-main')
WORK=ROOT/'experiments/p4_final_p1_training/p4_5_robot_reliability_server10h/index1'
OUT=WORK/'result/server'
PRIOR=ROOT/'experiments/p4_final_p1_training/p4_0_1_training_readiness_unblock/index1/result'
DATA=ROOT/'outputs/p4_0/formal6_merged'
FIELDS=['shoulder_pan.pos','shoulder_lift.pos','elbow_flex.pos','wrist_flex.pos','wrist_roll.pos','gripper.pos']
STOP=False


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)


def append(path,row):
    exists=path.exists()
    with path.open('a',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(row))
        if not exists:writer.writeheader()
        writer.writerow(row)


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore(value):
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['cpu']);torch.cuda.set_rng_state_all(value['cuda'])


class Mapped(Dataset):
    def __init__(self,ids):
        self.base=LeRobotDataset('QYyyyyyyy/formal6_merged',root=DATA,episodes=ids,
            delta_timestamps={'action':[i/30 for i in range(100)]},video_backend='torchcodec')
        names=self.base.meta.features['observation.state']['names']
        self.indices=[names.index(name) for name in FIELDS]
        assert self.base.meta.features['action']['names']==FIELDS

    def __len__(self):return len(self.base)

    def __getitem__(self,index):
        raw=self.base[index]
        raw['observation.state']=raw['observation.state'][self.indices]
        return {k:raw[k] for k in ['observation.state','observation.images.top','observation.images.gripper','action','action_is_pad']}


def clear_cache(_):
    video_utils._default_decoder_cache.clear()


def config():
    assert '/stock/src/' in inspect.getfile(ACTPolicy),'Stock source boundary violation'
    cfg=ACTConfig(input_features={'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(6,)),
        **{'observation.images.'+name:PolicyFeature(type=FeatureType.VISUAL,shape=(3,480,640)) for name in ['gripper','top']}},
        output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(6,))})
    assert not hasattr(cfg,'gripper_loss_weight') and cfg.optimizer_lr_backbone==1e-5
    return cfg


def statistics():
    stats=torch.load(ROOT/'experiments/p4_final_p1_training/p4_1_unattended_p1_pilot/index3/result/train90_statistics.pt',weights_only=False)
    stats['observation.state']={k:v[:6].clone() for k,v in stats['observation.state'].items()}
    assert stats['observation.state']['mean'].shape==(6,)
    return stats


def loader(dataset,seed,epoch,cursor=0):
    # Save the epoch and consumed cursor, reconstruct the same shuffle after a failure.
    gen=torch.Generator().manual_seed(seed+epoch)
    result=DataLoader(dataset,batch_size=TrainPipelineConfig.__dataclass_fields__['batch_size'].default,
        num_workers=4,shuffle=True,drop_last=False,pin_memory=True,generator=gen,worker_init_fn=clear_cache)
    iterator=iter(result)
    for _ in range(cursor):next(iterator)
    return iterator


def update(policy,opt,pre,raw,clip):
    policy.train();opt.zero_grad(set_to_none=True)
    loss,parts=policy(pre(raw));assert torch.isfinite(loss),'NONFINITE_LOSS'
    loss.backward();grad=torch.nn.utils.clip_grad_norm_(policy.parameters(),clip,error_if_nonfinite=True);opt.step()
    return float(loss.detach()),float(grad),parts


def smoke(folder,policy,opt,pre,raw,cfg,clip):
    batch=pre(raw)
    assert batch['observation.state'].shape==(8,6)
    assert batch['action'].shape==(8,100,6)
    assert all(torch.isfinite(v).all() for v in batch.values() if isinstance(v,torch.Tensor))
    update(policy,opt,pre,raw,clip)
    path=folder/'smoke_resume.pt'
    torch.save(dict(model=policy.state_dict(),optimizer=opt.state_dict(),rng=rng()),path)
    before=rng();first=update(policy,opt,pre,raw,clip)
    expected={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
    new=ACTPolicy(cfg).cuda();optimizer=cfg.get_optimizer_preset().build(new.get_optim_params())
    state=torch.load(path,map_location='cpu',weights_only=False)
    new.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);restore(state['rng'])
    second=update(new,optimizer,pre,raw,clip)
    difference=max(float((expected[k]-v.detach().cpu()).abs().max()) for k,v in new.state_dict().items())
    assert difference<=1e-6,(difference,first,second)
    # Export and reload an actual deployable stock policy, including processors' statistics.
    new.save_pretrained(folder/'smoke_pretrained')
    reloaded=ACTPolicy.from_pretrained(folder/'smoke_pretrained').cuda().eval()
    new.eval()
    with torch.inference_mode():
        a=new.predict_action_chunk(batch);b=reloaded.predict_action_chunk(batch)
    deploy_diff=float((a-b).abs().max());assert deploy_diff<=1e-6
    write(folder/'preflight.json',dict(status='PASS',real_train_batch=8,state_dim=6,action_shape=[8,100,6],
        next_update_max_weight_difference=difference,reloaded_chunk_max_difference=deploy_diff,
        discarded_updates=True,no_master=True,source=inspect.getfile(ACTPolicy)))


def checkpoint(folder,policy,opt,cfg,step,epoch,cursor):
    path=folder/'checkpoints'/f'{step:06d}';path.mkdir(parents=True,exist_ok=True)
    temp=path/'resume.tmp'
    torch.save(dict(model=policy.state_dict(),optimizer=opt.state_dict(),rng=rng(),step=step,epoch=epoch,cursor=cursor,
        config=cfg,split_sha256=sha(PRIOR/'split_freeze_manifest.json'),official_commit='8fff0fde7c79f23a93d845d1a50e985de01f8b8a'),temp)
    temp.replace(path/'resume.pt');cfg.save_pretrained(path/'pretrained_model')
    write(folder/'latest.json',dict(checkpoint=str(path/'resume.pt'),step=step,sha256=sha(path/'resume.pt')))
    append(folder/'checkpoint_manifest.csv',dict(step=step,path=str(path/'resume.pt'),sha256=sha(path/'resume.pt'),complete_resume=True))


@torch.inference_mode()
def evaluate(policy,pre,stats,run,step):
    ids=json.loads((PRIOR/'dev15_episode_ids.json').read_text());dataset=Mapped(ids)
    meta=pd.read_parquet(DATA/'meta/episodes/chunk-000/file-000.parquet',filters=[('episode_index','in',ids)])
    policy.eval();offset=0
    for ep in ids:
        length=int(meta[meta.episode_index==ep].iloc[0].length)
        for fraction in (0,.15,.3,.5,.7,.9):
            raw=dataset[offset+min(length-1,int(fraction*length))];batch=pre(torch.utils.data.default_collate([raw]))
            pred=policy.predict_action_chunk(batch);valid=~batch['action_is_pad'][0]
            err=(pred[0]-batch['action'][0]).abs()[valid];physical=err*stats['action']['std'].cuda()
            append(OUT/'dev15_model_comparison.csv',dict(run=run,step=step,episode=ep,progress_proxy=fraction,
                arm_normalized=float(err[:,:5].mean()),gripper_normalized=float(err[:,5].mean()),
                arm_raw=float(physical[:,:5].mean()),gripper_raw=float(physical[:,5].mean())))
        offset+=length


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--seed',type=int,default=1)
    p.add_argument('--benchmark',type=int,default=0);p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();folder=OUT/'runs'/a.run;folder.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,lambda *_:globals().__setitem__('STOP',True))
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    cfg=config();stats=statistics();pre,post=make_act_pre_post_processors(cfg,stats)
    policy=ACTPolicy(cfg).cuda();preset=cfg.get_optimizer_preset();opt=preset.build(policy.get_optim_params())
    cfg.save_pretrained(folder/'config');torch.save(stats,folder/'statistics.pt')
    write(folder/'identity.json',dict(run=a.run,seed=a.seed,config=json.loads((folder/'config/config.json').read_text()),
        batch_size=8,optimizer=vars(preset),source=inspect.getfile(ACTPolicy),
        observation_fields=FIELDS,method='OFFICIAL_STOCK_ACT',steps_target=80000,
        engineering_overrides=['dataset Train90/Dev15 adapter','seed','steps80000','no hub publication',
                               'explicit reproducible epoch shuffle seed+epoch','worker decoder-cache clearing']))
    dataset=Mapped(json.loads((PRIOR/'train90_episode_ids.json').read_text()))
    step=epoch=cursor=0
    if (folder/'latest.json').exists() and not a.benchmark:
        saved=torch.load(json.loads((folder/'latest.json').read_text())['checkpoint'],weights_only=False,map_location='cpu')
        assert saved['split_sha256']==sha(PRIOR/'split_freeze_manifest.json')
        policy.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        step,epoch,cursor=saved['step'],saved['epoch'],saved['cursor']
        iterator=loader(dataset,a.seed,epoch,cursor);restore(saved['rng'])
    else:iterator=loader(dataset,a.seed,epoch)
    if a.smoke:
        smoke(folder,policy,opt,pre,next(iterator),cfg,preset.grad_clip_norm)
        return
    start=time.monotonic();latencies=[];target=step+a.benchmark if a.benchmark else 80000
    while step<target and time.time()<a.deadline and not STOP:
        try:raw=next(iterator)
        except StopIteration:epoch+=1;cursor=0;iterator=loader(dataset,a.seed,epoch);raw=next(iterator)
        began=time.monotonic();loss,grad,parts=update(policy,opt,pre,raw,preset.grad_clip_norm)
        torch.cuda.synchronize();duration=time.monotonic()-began;latencies.append(duration);step+=1;cursor+=1
        if step==1 or step%100==0:
            row=dict(step=step,loss=loss,grad_norm=grad,epoch=epoch,cursor=cursor,elapsed_s=time.monotonic()-start)
            append(folder/'training.csv',row);print(a.run,row,flush=True)
            write(folder/'progress.json',dict(**row,pid=os.getpid(),peak_vram=torch.cuda.max_memory_allocated()))
        if not a.benchmark and step%20000==0:checkpoint(folder,policy,opt,cfg,step,epoch,cursor)
    if a.benchmark:
        elapsed=time.monotonic()-start
        write(folder/'benchmark.json',dict(run=a.run,steps=step,elapsed_s=elapsed,steps_per_s=step/elapsed,
            p95_update_s=float(np.percentile(latencies,95)),peak_vram=torch.cuda.max_memory_allocated(),batch_size=8))
        return
    checkpoint(folder,policy,opt,cfg,step,epoch,cursor)
    if step==80000:evaluate(policy,pre,stats,a.run,step)
    write(folder/'execution_complete.json',dict(status='DONE' if step==80000 else 'PARTIAL',step=step,elapsed_s=time.monotonic()-start))


if __name__=='__main__':main()
