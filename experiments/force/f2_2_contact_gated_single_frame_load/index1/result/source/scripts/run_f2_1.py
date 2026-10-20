"""F2.1 r1 matched training/evaluation and GPU engineering gates, no robot access."""
import argparse
import csv
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_STATE


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False))


def shared_hash(policy):
    h = hashlib.sha256()
    for key, value in policy.state_dict().items():
        if 'load_encoder' not in key and 'load_pos_embed' not in key:
            h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Experiment:
    def __init__(self, root, data, out):
        self.root, self.data, self.out = root, data, out
        self.spec = json.loads((root/'split_manifest.json').read_text())
        self.norm = json.loads((root/'normalization.json').read_text())
        self.cache = {}
        for ep in self.spec['train']+self.spec['dev']:
            with np.load(root/'arrays'/f'ep{ep:03d}.npz') as z:
                self.cache[ep] = {k:z[k].copy() for k in ['state','load_single','load_w15']}
        self.mean = {key:torch.tensor(self.norm[key]['mean'], device='cuda', dtype=torch.float32) for key in ['state','action']}
        self.std = {key:torch.tensor(self.norm[key]['std'], device='cuda', dtype=torch.float32).clamp_min(1e-8) for key in ['state','action']}
        self.image_mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1)
        self.image_std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)

    def config(self, variant):
        values=json.loads((self.root/'historical_config.json').read_text())
        values.pop('type',None)
        values['input_features']={
            OBS_STATE:PolicyFeature(type=FeatureType.STATE,shape=(6,)),
            **{key:PolicyFeature(type=FeatureType.VISUAL,shape=(3,480,640)) for key in ['observation.images.gripper','observation.images.top']}}
        values['output_features']={ACTION:PolicyFeature(type=FeatureType.ACTION,shape=(6,))}
        values['load_history_steps']={'A0':0,'A1':1,'A2':15}[variant]
        values['object_conditioning_prototype_path']=str(self.root/'cup_prototype_bank_v1.pt')
        return ACTConfig(**values)

    def policy(self, variant):
        torch.manual_seed(1)
        return ACTPolicy(self.config(variant)).to('cuda')

    def loader(self, role, *, workers=4):
        ds=LeRobotDataset('formal4',root=self.data,episodes=self.spec[role],video_backend='pyav',
            delta_timestamps={ACTION:[k/30 for k in range(100)]})
        generator=torch.Generator().manual_seed(1)
        return DataLoader(ds,batch_size=4,shuffle=role=='train',drop_last=role=='train',
            num_workers=workers,pin_memory=True,generator=generator,persistent_workers=workers>0)

    def batch(self, raw, variant):
        ids=raw['episode_index'].reshape(-1).tolist(); frames=raw['frame_index'].reshape(-1).tolist()
        state=torch.tensor(np.stack([self.cache[e]['state'][f] for e,f in zip(ids,frames)]),device='cuda')
        batch={OBS_STATE:(state-self.mean['state'])/self.std['state']}
        for key in ['observation.images.gripper','observation.images.top']:
            batch[key]=(raw[key].to('cuda',non_blocking=True)-self.image_mean)/self.image_std
        if variant!='A0':
            key='load_single' if variant=='A1' else 'load_w15'
            batch['observation.load']=torch.tensor(np.stack([self.cache[e][key][f] for e,f in zip(ids,frames)]),device='cuda')
        if ACTION in raw:
            batch[ACTION]=(raw[ACTION].to('cuda')-self.mean['action'])/self.std['action']
        if 'action_is_pad' in raw:
            batch['action_is_pad']=raw['action_is_pad'].to('cuda')
        return batch

    @torch.no_grad()
    def evaluate(self, policy, variant, loader, mechanisms=False):
        policy.eval(); sums={}; perturb=[]; horizons={h:np.zeros(3) for h in [1,5,10,30,100]}
        for raw in loader:
            batch=self.batch(raw,variant)
            targets=batch.pop(ACTION)
            chunks=policy.predict_action_chunk(batch)
            pred=chunks[:,0]; target=targets[:,0]
            for h,totals in horizons.items():
                mask=~batch['action_is_pad'][:,:h]
                delta=(chunks[:,:h]-targets[:,:h]).abs()[mask]
                totals += [float(delta[:,5].sum()),float(delta[:,:5].sum()),len(delta)]
            error=(pred-target).abs(); physical=error*self.std['action']
            ep=raw['episode_index'].reshape(-1).tolist()
            for i,e in enumerate(ep):
                s=sums.setdefault(e,np.zeros(5))
                s += [float(error[i,5]),float(physical[i,5]),float(physical[i,:5].mean()),float(error[i].mean()),1]
            if mechanisms and variant!='A0':
                original=batch['observation.load']
                for name,new in [('zero',torch.zeros_like(original)),('plus_0.1std',original+.1),('minus_0.1std',original-.1)]:
                    batch['observation.load']=new
                    delta=(policy.predict_action_chunk(batch)[:,0]-pred).abs()
                    perturb.append(dict(mode=name,gripper=float(delta[:,5].sum()),arm=float(delta[:,:5].mean(-1).sum()),n=len(ep)))
                batch['observation.load']=original
        totals=sum(sums.values())
        return dict(primary=float(totals[0]/totals[4]),physical_gripper_mae=float(totals[1]/totals[4]),
            physical_arm_mae=float(totals[2]/totals[4]),normalized_all_l1=float(totals[3]/totals[4]),frames=int(totals[4]),
            per_episode={str(e):dict(primary=float(s[0]/s[4]),frames=int(s[4])) for e,s in sums.items()},mechanisms=perturb,
            legal_horizons={str(h):dict(normalized_gripper_l1=float(s[0]/s[2]),normalized_arm_l1=float(s[1]/s[2]/5),valid_targets=int(s[2])) for h,s in horizons.items()})

    def smoke(self):
        raw=next(iter(self.loader('train',workers=0)))
        dev_raw=next(iter(self.loader('dev',workers=0)))
        result={}; hashes=[]
        for variant in ['A0','A1','A2']:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            policy=self.policy(variant); initial=shared_hash(policy); hashes.append(initial)
            batch=self.batch(raw,variant)
            assert batch[OBS_STATE].shape==(4,6) and batch[ACTION].shape==(4,100,6)
            start=time.perf_counter()
            policy.train(); loss,parts=policy(batch); assert torch.isfinite(loss)
            loss.backward()
            gradient_norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),10.,error_if_nonfinite=True)
            if variant!='A0':
                assert policy.model.load_encoder[0].weight.grad is not None
                assert torch.isfinite(policy.model.load_encoder[0].weight.grad).all()
                assert policy.model.load_encoder[0].weight.grad.abs().sum()>0
            assert policy.model.vae_encoder_robot_state_input_proj.in_features==6
            policy.eval()
            with torch.no_grad():
                before=policy.predict_action_chunk(batch)
                extra=dict(batch);extra['ignored_current']=torch.full((4,1),1e9,device='cuda')
                torch.testing.assert_close(before,policy.predict_action_chunk(extra),rtol=0,atol=0)
            ckpt=self.out/'smoke'/variant
            policy.save_pretrained(ckpt)
            reloaded=ACTPolicy.from_pretrained(ckpt).to('cuda').eval()
            with torch.no_grad():
                torch.testing.assert_close(before,reloaded.predict_action_chunk(batch),rtol=1e-5,atol=1e-6)
            optimizer=torch.optim.AdamW(policy.get_optim_params(),lr=1e-5,weight_decay=1e-4)
            optimizer.step()
            assert shared_hash(policy)!=initial,'Optimizer did not update common parameters'
            evaluation=self.evaluate(policy,variant,[dev_raw],mechanisms=True)
            assert evaluation['frames']==4 and np.isfinite(evaluation['primary'])
            assert set(evaluation['legal_horizons'])=={'1','5','10','30','100'}
            assert len(evaluation['mechanisms'])==(0 if variant=='A0' else 3)
            result[variant]=dict(shared_hash=initial,loss=float(loss.detach()),parameters=sum(p.numel() for p in policy.parameters()),
                trainable_parameters=sum(p.numel() for p in policy.parameters() if p.requires_grad),
                peak_vram_bytes=torch.cuda.max_memory_allocated(),gradient_norm=float(gradient_norm),seconds=time.perf_counter()-start,checks='real batch train/backward/optimizer update; 6D VAE; unused-current invariance; strict reload; Dev metrics/horizons/mechanisms')
            del policy,reloaded,batch,before,loss,optimizer;gc.collect()
        assert len(set(hashes))==1,'Common initialization mismatch'
        save(self.out/'smoke.json',dict(status='PASS',variants=result,shared_initialization_equal=True))

    def train(self, variant):
        if json.loads((self.out/'smoke.json').read_text())['status']!='PASS':
            raise RuntimeError('GPU gates not passed')
        if not (self.out/'historical_metric_reproduction.json').exists():
            raise RuntimeError('Full baseline primary metric reproduction required')
        out=self.out/variant
        out.mkdir(exist_ok=False)
        policy=self.policy(variant); optimizer=torch.optim.AdamW(policy.get_optim_params(),lr=1e-5,weight_decay=1e-4)
        loader=self.loader('train'); dev=self.loader('dev'); iterator=iter(loader)
        best=float('inf'); start=time.monotonic()
        with (out/'training.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['step','loss','l1_loss','kld_loss','grad_norm','elapsed_s']);writer.writeheader()
            for step in range(1,100001):
                try: raw=next(iterator)
                except StopIteration: iterator=iter(loader);raw=next(iterator)
                policy.train(); optimizer.zero_grad(set_to_none=True)
                loss,parts=policy(self.batch(raw,variant))
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                loss.backward(); norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),10.,error_if_nonfinite=True)
                optimizer.step()
                if step%100==0 or step==1:
                    row=dict(step=step,loss=float(loss.detach()),**parts,grad_norm=float(norm),elapsed_s=time.monotonic()-start)
                    writer.writerow(row);stream.flush();print(variant,row,flush=True)
                if step%20000==0:
                    checkpoint=out/'checkpoints'/f'{step:06d}'/'pretrained_model'
                    policy.save_pretrained(checkpoint)
                    shutil_normalization=self.root/'normalization.json'
                    (checkpoint/'f2_1_normalization.json').write_bytes(shutil_normalization.read_bytes())
                    score=self.evaluate(policy,variant,dev,mechanisms=True)
                    save(out/'evaluation'/f'{step:06d}.json',score)
                    torch.save(dict(step=step,optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),out/'latest_optimizer.pt')
                    if score['primary']<best:
                        best=score['primary'];save(out/'selected.json',dict(step=step,primary=best,checkpoint=str(checkpoint)))
            save(out/'complete.json',dict(steps=100000,elapsed_s=time.monotonic()-start,selected_primary=best))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--mode',choices=['smoke','train','historical'],required=True);ap.add_argument('--variant',choices=['A0','A1','A2'])
    ap.add_argument('--historical-checkpoint',type=Path)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    exp=Experiment(args.root,args.data,args.out)
    if args.mode=='smoke': exp.smoke()
    elif args.mode=='train': exp.train(args.variant)
    else:
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.configs.policies import PreTrainedConfig
        cfg=PreTrainedConfig.from_pretrained(args.historical_checkpoint)
        cfg.device='cuda'
        cfg.object_conditioning_prototype_path=str(args.root/'cup_prototype_bank_v1.pt')
        policy=ACTPolicy.from_pretrained(args.historical_checkpoint,config=cfg).to('cuda').eval()
        pre,_=make_pre_post_processors(policy_cfg=cfg,pretrained_path=str(args.historical_checkpoint))
        errors=[]
        for raw in exp.loader('dev'):
            batch=pre(raw)
            target=batch.pop(ACTION)
            target=target[:,0] if target.ndim==3 else target
            with torch.no_grad():
                pred=policy.predict_action_chunk(batch)[:,0]
            errors.extend((pred[:,5]-target[:,5]).abs().detach().cpu().tolist())
        score=float(np.mean(errors));reference=.05930798156050837
        assert len(errors)==5748 and abs(score-reference)<1e-4,(score,reference)
        save(args.out/'historical_metric_reproduction.json',dict(status='PASS',frames=len(errors),score=score,reference=reference,tolerance=1e-4))


if __name__=='__main__':main()
