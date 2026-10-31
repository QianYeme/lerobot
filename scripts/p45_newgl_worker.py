"""P4.5 historical two-stage New-GL reconstruction on formal6 Train90, seed1."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

FROZEN_SRC=Path('/root/autodl-tmp/lerobot-main/experiments/p4_final_p1_training/p4_5_robot_reliability_server10h/index1/newgl_code/src')
if FROZEN_SRC.exists():sys.path.insert(0,str(FROZEN_SRC))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors

import p45_vanilla_worker as common

ROOT,WORK,OUT,PRIOR,DATA=common.ROOT,common.WORK,common.OUT,common.PRIOR,common.DATA
RAW={};STOP=False
_append=common.append


def locked_append(path,row):
    with path.with_suffix(path.suffix+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);_append(path,row)


common.append=locked_append


def checkpoint(folder,policy,opt,cfg,step,epoch,cursor):
    path=folder/'checkpoints'/f'{step:06d}';path.mkdir(parents=True,exist_ok=True)
    temp=path/'resume.tmp'
    torch.save(dict(model=policy.state_dict(),optimizer=opt.state_dict(),rng=common.rng(),step=step,epoch=epoch,cursor=cursor,
        config=cfg,split_sha256=common.sha(PRIOR/'split_freeze_manifest.json'),
        source_kind='PROJECT_NEWGL_RECONSTRUCTED',worker_sha256=common.sha(Path(__file__))),temp)
    temp.replace(path/'resume.pt');cfg.save_pretrained(path/'pretrained_model')
    common.write(folder/'latest.json',dict(checkpoint=str(path/'resume.pt'),step=step,sha256=common.sha(path/'resume.pt')))
    common.append(folder/'checkpoint_manifest.csv',dict(step=step,path=str(path/'resume.pt'),sha256=common.sha(path/'resume.pt'),complete_resume=True))


class Mapped(common.Mapped):
    def __init__(self,ids):
        super().__init__(ids)
        names=self.base.meta.features['observation.state']['names'];self.load_index=names.index('gripper.load')

    def __getitem__(self,index):
        raw=self.base[index]
        result={k:raw[k] for k in ['action','action_is_pad','observation.images.top','observation.images.gripper','episode_index','frame_index']}
        result['observation.load']=raw['observation.state'][self.load_index:self.load_index+1]
        result['observation.state']=raw['observation.state'][self.indices]
        return result


def make(fixed):
    values=json.loads((OUT/'newgl_historical_config.json').read_text())
    values['object_conditioning_prototype_path']=str(ROOT/'experiments/p4_final_p1_training/p4_1_unattended_p1_pilot/index3/result/formal6_train90_prototypes.pt')
    folder=OUT/'ng_config';folder.mkdir(exist_ok=True)
    # Config is identical in the two arms; fixed input is a runtime boundary hook only.
    path=folder/('fixed' if fixed else 'legacy');path.mkdir(exist_ok=True)
    (path/'config.json').write_text(json.dumps(values))
    cfg=PreTrainedConfig.from_pretrained(path)
    assert cfg.load_history_steps==1 and tuple(cfg.robot_state_feature.shape)==(6,)
    policy=ACTPolicy(cfg).cuda()
    if fixed:
        def raw_top(_module,args,kwargs):return (args[0],RAW['top'],*args[2:]),kwargs
        policy.model.object_conditioning.register_forward_pre_hook(raw_top,with_kwargs=True)
    return policy,cfg


def processors(cfg):
    stats=common.statistics();pre,post=make_act_pre_post_processors(cfg,stats)
    original=torch.load(ROOT/'experiments/p4_final_p1_training/p4_1_unattended_p1_pilot/index3/result/train90_statistics.pt',weights_only=False)
    load_mean=original['observation.state']['mean'][6].cuda();load_std=original['observation.state']['std'][6].cuda()
    def prepare(raw):
        top=raw['observation.images.top'];RAW['top']=(top[None] if top.ndim==3 else top).cuda()
        batch=pre(raw)
        batch['observation.load']=((batch['observation.load']-load_mean)/load_std).reshape(batch['observation.state'].shape[0],1)
        return batch
    return prepare,post,stats


def samples(ids):
    dataset=Mapped(ids)
    meta=pd.read_parquet(DATA/'meta/episodes/chunk-000/file-000.parquet',filters=[('episode_index','in',ids)])
    offset=0
    for ep in ids:
        length=int(meta[meta.episode_index==ep].iloc[0].length)
        for fraction in (0,.15,.3,.5,.7,.9):
            yield ep,fraction,dataset[offset+min(length-1,int(fraction*length))]
        offset+=length


@torch.inference_mode()
def evaluate(policy,pre,stats,run,step):
    policy.eval();rows=[]
    for ep,fraction,raw in samples(json.loads((PRIOR/'dev15_episode_ids.json').read_text())):
        batch=pre(torch.utils.data.default_collate([raw]));pred=policy.predict_action_chunk(batch);target=batch['action'];assert target.ndim==3 and target.shape==pred.shape
        valid=~batch['action_is_pad'][0]
        err=(pred[0]-target[0]).abs()[valid];physical=err*stats['action']['std'].cuda()
        gate=policy.model.object_conditioning.gate;alpha=gate.detach().clone()
        gate.zero_()
        try:off=policy.predict_action_chunk(batch)
        finally:gate.copy_(alpha)
        no_load=dict(batch);no_load['observation.load']=torch.zeros_like(batch['observation.load'])
        neutral=policy.predict_action_chunk(no_load)
        row=dict(run=run,step=step,episode=ep,progress_proxy=fraction,
            arm_normalized=float(err[:,:5].mean()),gripper_normalized=float(err[:,5].mean()),
            arm_raw=float(physical[:,:5].mean()),gripper_raw=float(physical[:,5].mean()),
            h0_gripper_normalized=float((pred[0,0,5]-target[0,0,5]).abs()))
        common.append(OUT/'newgl_dev15_candidates.csv',row);rows.append(row)
        common.append(OUT/'object_path_diagnostics.csv',dict(run=run,step=step,episode=ep,progress_proxy=fraction,
            object_on_off=float((pred-off).abs().mean()),load_zero_difference=float((pred-neutral).abs().mean()),gate=float(alpha)))
    return float(np.mean([r['h0_gripper_normalized'] for r in rows]))


def preflight():
    dataset=Mapped(json.loads((PRIOR/'train90_episode_ids.json').read_text()))
    raw=next(iter(DataLoader(dataset,batch_size=4,num_workers=0)))
    records={};initials=[]
    for fixed in [False,True]:
        random.seed(1);np.random.seed(1);torch.manual_seed(1);torch.cuda.manual_seed_all(1)
        policy,cfg=make(fixed);pre,post,stats=processors(cfg);batch=pre(raw)
        h=__import__('hashlib').sha256()
        for name,v in policy.state_dict().items():h.update(name.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        initials.append(h.hexdigest())
        dino=policy.model.object_conditioning.dino.model;original=dino.forward_features;seen={}
        def capture(image,*args,**kwargs):
            seen['pixels']=image.detach().clone();return original(image,*args,**kwargs)
        dino.forward_features=capture
        policy.eval()
        with torch.inference_mode():policy.predict_action_chunk(batch)
        image=RAW['top'] if fixed else batch['observation.images.top']
        expected=torch.nn.functional.interpolate(image,(224,224),mode='bilinear',align_corners=False)
        mean=expected.new_tensor([.485,.456,.406]).view(1,3,1,1);std=expected.new_tensor([.229,.224,.225]).view(1,3,1,1)
        difference=float((seen['pixels']-(expected-mean)/std).abs().max());assert difference<=1e-5
        dino.forward_features=original
        opt=torch.optim.AdamW(policy.get_optim_params(),lr=1e-5,weight_decay=1e-4)
        loss,grad,_=common.update(policy,opt,pre,raw,10.)
        folder=OUT/'newgl_smoke_native_numerics'/('fixed' if fixed else 'legacy');folder.mkdir(parents=True,exist_ok=True)
        checkpoint(folder,policy,opt,cfg,1,0,1)
        saved=torch.load(json.loads((folder/'latest.json').read_text())['checkpoint'],weights_only=False,map_location='cpu')
        first=common.update(policy,opt,pre,raw,10.)
        expected_weights={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
        clone,_=make(fixed);clone.load_state_dict(saved['model']);optimizer=torch.optim.AdamW(clone.get_optim_params(),lr=1e-5,weight_decay=1e-4)
        optimizer.load_state_dict(saved['optimizer']);common.restore(saved['rng']);common.update(clone,optimizer,pre,raw,10.)
        reload_difference=max(float((expected_weights[k]-v.detach().cpu()).abs().max()) for k,v in clone.state_dict().items())
        assert reload_difference<=1e-6
        records['fixed' if fixed else 'legacy']=dict(real_train_loss=loss,grad_norm=grad,dino_input_max_difference=difference,
            optimizer_next_update_max_difference=reload_difference,load_shape=list(batch['observation.load'].shape),state_shape=list(batch['observation.state'].shape))
        del policy,clone,opt,optimizer;torch.cuda.empty_cache()
    assert initials[0]==initials[1]
    common.write(OUT/'preprocessing_probe.json',dict(status='PASS',branches=records,initial_state_hash=initials[0],same_initial_weights=True))


def ensure_slots(deadline):
    lock=(OUT/'ng_benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    path=OUT/'ng_parallel_decision.json'
    if not path.exists():
        speeds=[]
        for count in [1,2]:
            jobs=[];began=time.time()
            for i in range(count):
                name=f"NG-{'F' if i else 'L'}-BENCH_{count}_{i}";log=(OUT/'training_logs'/(name+'.log')).open('a')
                p=subprocess.Popen([sys.executable,__file__,'--run',name,'--stage','global','--benchmark','100','--deadline',str(deadline)],stdout=log,stderr=subprocess.STDOUT)
                log.close();jobs.append((name,p))
            codes=[p.wait() for _,p in jobs];elapsed=time.time()-began
            completed=[json.loads((OUT/'runs'/name/'benchmark.json').read_text())['steps'] for name,_ in jobs] if all(code==0 for code in codes) else []
            speed=sum(completed)/elapsed if len(completed)==count and all(value==100 for value in completed) else 0
            common.append(OUT/'parallel_ng_benchmark.csv',dict(workers=count,steps_per_s=speed,walltime_s=elapsed,codes=str(codes)))
            speeds.append(speed)
        assert speeds[0]>0
        common.write(path,dict(workers=2 if speeds[1]>=1.1*speeds[0] else 1,speeds=speeds))
    workers=json.loads(path.read_text())['workers'];fcntl.flock(lock,fcntl.LOCK_UN)
    while time.time()<deadline:
        for index in range(workers):
            slot=(OUT/f'ng_slot_{index}.lock').open('a')
            try:fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB);return slot
            except BlockingIOError:slot.close()
        time.sleep(2)
    raise TimeoutError('NG slot deadline')


def global_train(a):
    slot=ensure_slots(a.deadline) if not a.benchmark else None
    fixed=a.run.startswith('NG-F');policy,cfg=make(fixed);pre,post,stats=processors(cfg)
    folder=OUT/'runs'/a.run;folder.mkdir(parents=True,exist_ok=True)
    cfg.save_pretrained(folder/'config');torch.save(stats,folder/'statistics.pt')
    common.write(folder/'identity.json',dict(run=a.run,seed=1,global_steps=100000,batch_size=4,load_history_steps=1,
        state_fields=common.FIELDS,extra_input='separate normalized gripper.load token; no current/master',
        fixed_raw_top=fixed,worker_sha256=common.sha(Path(__file__)),configuration=json.loads((folder/'config/config.json').read_text()),
        historical_protocol='F4.1 train_global100k; Dev h0 gripper best among20/40/60/80/100k',
        historical_exact_reproduction=False,reason='New formal6 task/split/Train90 prototypes; documented epoch shuffle adapter'))
    opt=torch.optim.AdamW(policy.get_optim_params(),lr=1e-5,weight_decay=1e-4)
    dataset=Mapped(json.loads((PRIOR/'train90_episode_ids.json').read_text()));step=epoch=cursor=0
    if (folder/'latest.json').exists() and not a.benchmark:
        saved=torch.load(json.loads((folder/'latest.json').read_text())['checkpoint'],weights_only=False,map_location='cpu')
        assert saved['split_sha256']==common.sha(PRIOR/'split_freeze_manifest.json')
        policy.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);step,epoch,cursor=saved['step'],saved['epoch'],saved['cursor']
    else:saved=None
    def batches(epoch,cursor):
        gen=torch.Generator().manual_seed(a.seed+epoch)
        it=iter(DataLoader(dataset,batch_size=4,shuffle=True,num_workers=4,pin_memory=True,drop_last=True,
            generator=gen,worker_init_fn=common.clear_cache))
        for _ in range(cursor):next(it)
        return it
    iterator=batches(epoch,cursor)
    if saved:common.restore(saved['rng'])
    begin=time.time();target=a.benchmark or 100000
    while step<target and time.time()<a.deadline and not STOP:
        try:raw=next(iterator)
        except StopIteration:epoch+=1;cursor=0;iterator=batches(epoch,cursor);raw=next(iterator)
        loss,grad,_=common.update(policy,opt,pre,raw,10.);step+=1;cursor+=1
        if step==1 or step%100==0:
            row=dict(step=step,loss=loss,grad_norm=grad,elapsed_s=time.time()-begin)
            common.append(folder/'training.csv',row);common.write(folder/'progress.json',dict(**row,pid=os.getpid()));print(a.run,row,flush=True)
        if not a.benchmark and step%20000==0:
            checkpoint(folder,policy,opt,cfg,step,epoch,cursor)
            primary=evaluate(policy,pre,stats,a.run,step)
            common.append(folder/'selection.csv',dict(step=step,primary=primary,checkpoint=str(folder/'checkpoints'/f'{step:06d}'/'resume.pt')))
    if a.benchmark:
        common.write(folder/'benchmark.json',dict(steps=step,elapsed_s=time.time()-begin));return
    if not (folder/'latest.json').exists() or json.loads((folder/'latest.json').read_text())['step']!=step:
        checkpoint(folder,policy,opt,cfg,step,epoch,cursor)
    if step==100000:
        selected=min(pd.read_csv(folder/'selection.csv').to_dict('records'),key=lambda r:(r['primary'],r['step']))
        common.write(folder/'selected_global.json',selected)
    common.write(folder/'execution_complete.json',dict(status='DONE' if step==100000 else 'PARTIAL',step=step,fixed=fixed,elapsed_s=time.time()-begin))


class Local(nn.Module):
    def __init__(self,bound):
        super().__init__();self.bound=bound
        self.input=nn.Linear(3,32);self.hidden=nn.Linear(32,32);self.output=nn.Linear(32,1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1);self.input.reset_parameters()
            torch.manual_seed(2);self.hidden.reset_parameters();self.output.reset_parameters()
            nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,x):return self.bound*torch.tanh(self.output(torch.nn.functional.gelu(self.hidden(torch.nn.functional.gelu(self.input(x)))))).squeeze(-1)


@torch.inference_mode()
def replay(policy,pre,stats,train):
    policy.eval();ids=json.loads((PRIOR/('train90_episode_ids.json' if train else 'dev15_episode_ids.json')).read_text())
    if train:
        ds=Mapped(ids);items=((None,None,raw) for raw in DataLoader(ds,batch_size=4,shuffle=False,num_workers=4,worker_init_fn=common.clear_cache))
    else:items=samples(ids)
    records={k:[] for k in ['nominal','target','position','load','episode','demo_native']}
    count=0
    for _,_,raw in items:
        if not train:raw=torch.utils.data.default_collate([raw])
        if STOP:raise InterruptedError('Replay interrupted; no partial cache substituted')
        batch=pre(raw);pred=policy.predict_action_chunk(batch)[:,0].cpu().numpy()
        records['nominal'].append(pred);records['target'].append(batch['action'][:,0].cpu().numpy())
        records['position'].append(batch['observation.state'][:,5].cpu().numpy()*float(stats['observation.state']['std'][5])+float(stats['observation.state']['mean'][5]))
        records['load'].append(batch['observation.load'][:,0].cpu().numpy())
        records['episode'].append(raw['episode_index'].reshape(-1).numpy().copy())
        action=raw['action'];records['demo_native'].append((action[0,5].reshape(1) if action.ndim==2 else action[:,0,5]).numpy().copy())
        count+=len(pred)
        if count%800==0:print('REPLAY','train' if train else 'dev',count,flush=True)
    return {k:np.concatenate(v) for k,v in records.items()}


def local_train(a):
    folder=OUT/'runs'/a.run;folder.mkdir(parents=True,exist_ok=True)
    parent=OUT/'runs'/a.run.replace('-L-L','-L-G').replace('-F-L','-F-G')
    selected=json.loads((parent/'selected_global.json').read_text());fixed=a.run.startswith('NG-F')
    policy,cfg=make(fixed);state=torch.load(selected['checkpoint'],weights_only=False,map_location='cpu');policy.load_state_dict(state['model']);del state
    policy.eval().requires_grad_(False);pre,post,stats=processors(cfg)
    for role,train in [('train',True),('dev',False)]:
        cache=folder/(role+'_cache.npz')
        if not cache.exists():
            values=replay(policy,pre,stats,train);np.savez_compressed(cache,**values)
    train=dict(np.load(folder/'train_cache.npz'));dev=dict(np.load(folder/'dev_cache.npz'))
    std=float(stats['action']['std'][5]);mean=float(stats['action']['mean'][5])
    demonstration=train['demo_native'];bound=.05*float(demonstration.max()-demonstration.min());assert bound>0
    nominal=train['nominal'][:,5]*std+mean
    normal={key:dict(mean=float(values.mean(dtype=np.float64)),std=max(float(values.std(dtype=np.float64)),1e-8)) for key,values in [('position',train['position']),('nominal',nominal)]}
    def inputs(d):
        pos=(d['position']-normal['position']['mean'])/normal['position']['std']
        nom=(d['nominal'][:,5]*std+mean-normal['nominal']['mean'])/normal['nominal']['std']
        return torch.tensor(np.stack([d['load'],pos,nom],1).astype(np.float32),device='cuda')
    x=inputs(train);dx=inputs(dev);target=torch.tensor(np.clip(demonstration-nominal,-bound,bound)/bound,device='cuda')
    local=Local(bound).cuda();opt=torch.optim.Adam(local.parameters(),lr=.001);generator=torch.Generator(device='cuda').manual_seed(1)
    if not (folder/'latest.json').exists():
        # Disposable local test on actual Train90 replay; the formal model stays untouched.
        probe=copy.deepcopy(local);probe_opt=torch.optim.Adam(probe.parameters(),lr=.001)
        assert sum(p.numel() for p in probe.parameters())==1217
        assert float(probe(dx).abs().max())==0.,'Local OFF must equal initial zero residual'
        sx=x[:4];sy=target[:4]
        for _ in range(2):
            probe_opt.zero_grad();loss=((probe(sx)/bound-sy)**2).mean();loss.backward();probe_opt.step()
        smoke=folder/'local_real_smoke.pt'
        torch.save(dict(model=probe.state_dict(),optimizer=probe_opt.state_dict()),smoke)
        clone=copy.deepcopy(local);clone_opt=torch.optim.Adam(clone.parameters(),lr=.001)
        saved=torch.load(smoke,weights_only=False);clone.load_state_dict(saved['model']);clone_opt.load_state_dict(saved['optimizer'])
        for model,optimizer in [(probe,probe_opt),(clone,clone_opt)]:
            optimizer.zero_grad();loss=((model(sx)/bound-sy)**2).mean();assert torch.isfinite(loss)
            loss.backward();optimizer.step()
        difference=max(float((v-clone.state_dict()[k]).abs().max()) for k,v in probe.state_dict().items())
        assert difference<=1e-6
        assert all(p.grad is None and not p.requires_grad for p in policy.parameters())
        assert float(probe(dx).abs().max())<=bound+1e-6
        common.write(folder/'local_preflight.json',dict(status='PASS',real_train_examples=4,parameters=1217,
            optimizer_reload_max_difference=difference,global_frozen=True,initial_residual_zero=True,formal_updates_not_consumed=True))
        del probe,clone,probe_opt,clone_opt,saved
    step=0
    if (folder/'latest.json').exists():
        state=torch.load(json.loads((folder/'latest.json').read_text())['checkpoint'],weights_only=False)
        assert state['global_sha256']==common.sha(Path(selected['checkpoint']))
        local.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);generator.set_state(state['generator']);common.restore(state['rng']);step=state['step']
    begin=time.time()
    while step<10000 and time.time()<a.deadline and not STOP:
        idx=torch.randint(len(x),(256,),device='cuda',generator=generator)
        opt.zero_grad(set_to_none=True);loss=((local(x[idx])/bound-target[idx])**2).mean();assert torch.isfinite(loss)
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(local.parameters(),float('inf'),error_if_nonfinite=True);opt.step();step+=1
        if step%100==0:common.append(folder/'training.csv',dict(step=step,loss=float(loss.detach()),grad_norm=float(grad),elapsed_s=time.time()-begin))
        if step%1000==0 or STOP:
            ckpt=folder/f'{step:06d}.pt';tmp=ckpt.with_suffix('.tmp')
            torch.save(dict(model=local.state_dict(),optimizer=opt.state_dict(),generator=generator.get_state(),rng=common.rng(),step=step,
                global_checkpoint=selected['checkpoint'],global_sha256=common.sha(Path(selected['checkpoint'])),normalization=normal,bound=bound),tmp);tmp.replace(ckpt)
            common.write(folder/'latest.json',dict(checkpoint=str(ckpt),step=step))
            with torch.inference_mode():delta=local(dx).cpu().numpy()
            for mode in ['OFF','ON']:
                pred=dev['nominal'].copy()
                if mode=='ON':pred[:,5]+=delta/std
                assert np.array_equal(pred[:,:5],dev['nominal'][:,:5]),'Local must never alter arm channels'
                error=np.abs(pred-dev['target']);physical=error*stats['action']['std'].numpy()
                for ep in np.unique(dev['episode']):
                    mask=dev['episode']==ep
                    common.append(OUT/'newgl_global_local_ablation.csv',dict(run=a.run,step=step,global_step=selected['step'],mode=mode,episode=int(ep),
                        h0_gripper_normalized=float(error[mask,5].mean()),gripper_raw=float(physical[mask,5].mean()),arm_raw=float(physical[mask,:5].mean()),
                        residual_mean_abs=float(np.abs(delta[mask]).mean()) if mode=='ON' else 0.,global_frozen=True))
    if step and (not (folder/'latest.json').exists() or json.loads((folder/'latest.json').read_text())['step']!=step):
        ckpt=folder/f'{step:06d}.pt';tmp=ckpt.with_suffix('.tmp')
        torch.save(dict(model=local.state_dict(),optimizer=opt.state_dict(),generator=generator.get_state(),rng=common.rng(),step=step,
            global_checkpoint=selected['checkpoint'],global_sha256=common.sha(Path(selected['checkpoint'])),normalization=normal,bound=bound),tmp)
        tmp.replace(ckpt);common.write(folder/'latest.json',dict(checkpoint=str(ckpt),step=step))
    common.write(folder/'execution_complete.json',dict(status='DONE' if step==10000 else 'PARTIAL',step=step,global_checkpoint=selected,bound=bound,
        normalization=normal,global_frozen=True,elapsed_s=time.time()-begin))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--seed',type=int,default=1)
    p.add_argument('--deadline',type=float,required=True);p.add_argument('--stage',choices=['preflight','global','local'],required=True);p.add_argument('--benchmark',type=int,default=0)
    a=p.parse_args();assert a.seed==1
    signal.signal(signal.SIGTERM,lambda *_:globals().__setitem__('STOP',True))
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;torch.backends.cudnn.deterministic=False
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    random.seed(1);np.random.seed(1);torch.manual_seed(1);torch.cuda.manual_seed_all(1)
    if a.stage=='preflight':preflight()
    elif a.stage=='global':global_train(a)
    else:local_train(a)


if __name__=='__main__':main()
