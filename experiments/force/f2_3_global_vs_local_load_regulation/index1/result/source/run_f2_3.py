"""F2.3 frozen-global replay and matched bounded local residual training."""
import argparse
import csv
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from run_f2_1 import Experiment, save
from lerobot.policies.act.modeling_act import ACTPolicy


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LocalResidual(nn.Module):
    def __init__(self, use_load, bound):
        super().__init__()
        self.bound = bound
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1)
            self.input = nn.Linear(3 if use_load else 2, 32)
            torch.manual_seed(2)
            self.hidden = nn.Linear(32, 32)
            self.output = nn.Linear(32, 1)
            nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, inputs):
        value = torch.nn.functional.gelu(self.input(inputs))
        value = torch.nn.functional.gelu(self.hidden(value))
        return self.bound * torch.tanh(self.output(value)).squeeze(-1)


def metrics(prediction, target, std, episodes):
    error = np.abs(prediction-target)
    physical = error*std
    result = dict(primary=float(error[:,5].mean()), physical_gripper_mae=float(physical[:,5].mean()),
        physical_arm_mae=float(physical[:,:5].mean()), normalized_all_l1=float(error.mean()),frames=len(error))
    result['per_episode'] = {str(ep):metrics(prediction[episodes==ep],target[episodes==ep],std,np.zeros(sum(episodes==ep),dtype=int))
                             for ep in np.unique(episodes)} if len(np.unique(episodes))>1 else {}
    return result


def data_inputs(arrays, normal, use_load):
    position = (arrays['position']-normal['position']['mean'])/normal['position']['std']
    nominal = (arrays['nominal_native']-normal['nominal']['mean'])/normal['nominal']['std']
    fields = [arrays['load'], position, nominal] if use_load else [position,nominal]
    return np.stack(fields,axis=1).astype(np.float32)


def cache(base):
    out=base/'result'; inputs=base/'inputs'
    identities=json.loads((inputs/'baseline_identity.json').read_text())
    config=json.loads((inputs/'residual_config.json').read_text())
    assert digest(inputs/'decision.md')==config['contract_sha256']
    exp=Experiment(inputs,Path('/root/autodl-tmp/lerobot/formal4_data/kind_merged_nomaster'),out)
    models={}
    for name, variant in [('C1_global','A0'),('C0','A1')]:
        identity=identities[name]; checkpoint=Path(identity['checkpoint'])
        assert digest(checkpoint/'model.safetensors')==identity['model_sha256'], 'BLOCKED_GLOBAL_CHECKPOINT_IDENTITY'
        assert (checkpoint/'f2_1_normalization.json').read_bytes()==(inputs/'normalization.json').read_bytes()
        policy=ACTPolicy.from_pretrained(checkpoint).cuda().eval()
        assert tuple(policy.config.robot_state_feature.shape)==(6,)
        assert policy.config.load_history_steps==(0 if variant=='A0' else 1)
        assert not getattr(policy.config,'load_feature_gating',False)
        policy.requires_grad_(False)
        assert all(not p.requires_grad for p in policy.parameters())
        models[name]=policy
        identity['config']=json.loads((checkpoint/'config.json').read_text())
        identity['training_config']=dict(steps=100000,seed=1,batch_size=4,checkpoint_every=20000,
            origin='F2.1-r1; A0 once authorized retry disclosed',split=exp.spec)
        identity['parameters']=sum(p.numel() for p in policy.parameters())
    save(out/'baseline_identity.json',identities)
    save(out/'global_checkpoint_identity.json',identities['C1_global'])
    for role in ['train','dev']:
        path=base/'cache'/f'{role}.npz'
        if path.exists():raise RuntimeError('Do not overwrite existing replay cache')
        path.parent.mkdir(exist_ok=True)
        records={key:[] for key in ['episode','frame','position','load','target','nominal','c0','demo_native']}
        ds=exp.loader(role,workers=0).dataset
        loader=DataLoader(ds,batch_size=4,shuffle=False,drop_last=False,num_workers=4,pin_memory=True)
        with torch.inference_mode():
            for raw in loader:
                batch=exp.batch(raw,'A0'); target=batch.pop('action')[:,0]
                assert batch['observation.state'].shape[1]==6 and 'observation.load' not in batch
                nominal=models['C1_global'].predict_action_chunk(batch)[:,0]
                ids=raw['episode_index'].reshape(-1).tolist(); frames=raw['frame_index'].reshape(-1).tolist()
                load=np.asarray([exp.cache[e]['load_single'][f,0] for e,f in zip(ids,frames)],dtype=np.float32)
                batch['observation.load']=torch.tensor(load[:,None],device='cuda')
                c0=models['C0'].predict_action_chunk(batch)[:,0]
                for key,value in [('episode',ids),('frame',frames),('load',load),
                    ('position',[exp.cache[e]['state'][f,5] for e,f in zip(ids,frames)]),
                    ('target',target.cpu().numpy()),('nominal',nominal.cpu().numpy()),('c0',c0.cpu().numpy()),
                    ('demo_native',raw['action'][:,0,5].numpy())]:
                    records[key].append(np.asarray(value))
        arrays={key:np.concatenate(value) for key,value in records.items()}
        expected=sum(len(exp.cache[e]['state']) for e in exp.spec[role])
        assert len(arrays['episode'])==expected and set(arrays['episode'])==set(exp.spec[role])
        assert len(set(zip(arrays['episode'],arrays['frame'])))==expected
        assert all(np.isfinite(v).all() for v in arrays.values())
        arrays['nominal_native']=arrays['nominal'][:,5]*exp.norm['action']['std'][5]+exp.norm['action']['mean'][5]
        # Verify original demo range using canonical raw arrays, not normalization roundoff.
        raw_demo=[]
        for ep in exp.spec[role]:
            with np.load(inputs/'arrays'/f'ep{ep:03d}.npz') as z:raw_demo.extend(z['action'][:,5].tolist())
        if role=='train':
            assert abs(.05*(max(raw_demo)-min(raw_demo))-config['delta_max'])<1e-10
        arrays['raw_residual']=arrays['demo_native']-arrays['nominal_native']
        arrays['residual_target']=np.clip(arrays['raw_residual'],-config['delta_max'],config['delta_max'])
        np.savez_compressed(path,**arrays)
        print('REPLAY',role,expected,flush=True)
        if role=='dev':
            for name,key in [('C1_global','nominal'),('C0','c0')]:
                score=metrics(arrays[key],arrays['target'],np.asarray(exp.norm['action']['std']),arrays['episode'])
                assert abs(score['primary']-identities[name]['primary'])<1e-4, (name,score['primary'])
                identities[name]['reproduced_primary']=score['primary']
    for name in models:
        assert all(p.grad is None and not p.requires_grad for p in models[name].parameters())
        assert digest(Path(identities[name]['checkpoint'])/'model.safetensors')==identities[name]['model_sha256']
    train=dict(np.load(base/'cache/train.npz'))
    normal={key:dict(mean=float(train[field].mean(dtype=np.float64)),std=max(float(train[field].std(dtype=np.float64)),1e-8))
            for key,field in [('position','position'),('nominal','nominal_native')]}
    normal['load']=exp.norm['load']; normal['target']='residual_target / delta_max, no centering'
    save(out/'local_normalization.json',normal)
    raw=train['raw_residual']; bound=config['delta_max']
    save(out/'residual_target_stats.json',dict(frames=len(raw),positive_clipped_fraction=float((raw>bound).mean()),
        negative_clipped_fraction=float((raw<-bound).mean()),total_saturation_fraction=float((np.abs(raw)>bound).mean()),
        raw_mean=float(raw.mean()),raw_abs_mean=float(np.abs(raw).mean()),delta_max=bound,scope='Train36 before local training'))
    save(out/'baseline_identity.json',identities)
    save(out/'global_checkpoint_identity.json',identities['C1_global'])
    save(out/'replay_complete.json',dict(status='PASS',Train_frames=17243,Dev_frames=5748,Final_reads=0,
        metric_reproduction_tolerance=1e-4,all_global_parameters_frozen=True))


def train_local(base):
    out=base/'result'; cfg=json.loads((base/'inputs/residual_config.json').read_text())
    normal=json.loads((out/'local_normalization.json').read_text())
    norm=json.loads((base/'inputs/normalization.json').read_text())
    assert json.loads((out/'replay_complete.json').read_text())['status']=='PASS'
    train=dict(np.load(base/'cache/train.npz')); dev=dict(np.load(base/'cache/dev.npz'))
    bound=cfg['delta_max']; target=torch.tensor(train['residual_target']/bound,device='cuda',dtype=torch.float32)
    std=norm['action']['std'][5]
    for variant in ['C1-no-load','C1-load']:
        directory=out/variant; directory.mkdir(exist_ok=False)
        use_load=variant=='C1-load'; model=LocalResidual(use_load,bound).cuda()
        x=torch.tensor(data_inputs(train,normal,use_load),device='cuda')
        xd=torch.tensor(data_inputs(dev,normal,use_load),device='cuda')
        # Disposable real-data gradient smoke; formal model is freshly initialized afterward.
        smoke=LocalResidual(use_load,bound).cuda(); opt=torch.optim.Adam(smoke.parameters(),lr=cfg['learning_rate'])
        assert torch.count_nonzero(smoke(x[:4]))==0
        for _ in range(2):
            opt.zero_grad(); loss=((smoke(x[:4])/bound-target[:4])**2).mean();loss.backward();opt.step()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in smoke.parameters())
        assert smoke.input.weight.grad.abs().sum()>0
        assert smoke(x[:4]).abs().max()<=bound
        del smoke,opt
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
        generator=torch.Generator(device='cuda').manual_seed(cfg['seed'])
        best=float('inf'); started=time.monotonic(); checkpoints=[]
        with (directory/'training.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['step','loss','grad_norm','elapsed_s']);writer.writeheader()
            for step in range(1,cfg['steps']+1):
                index=torch.randint(len(x),(cfg['batch_size'],),device='cuda',generator=generator)
                optimizer.zero_grad();loss=((model(x[index])/bound-target[index])**2).mean()
                assert torch.isfinite(loss);loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
                optimizer.step()
                if step==1 or step%100==0:
                    row=dict(step=step,loss=float(loss.detach()),grad_norm=float(grad),elapsed_s=time.monotonic()-started)
                    writer.writerow(row);stream.flush();print(variant,row,flush=True)
                if step%cfg['checkpoint_every']==0:
                    model.eval()
                    with torch.no_grad():delta=model(xd).cpu().numpy()
                    pred=dev['nominal'].copy();pred[:,5]+=delta/std
                    score=float(np.abs(pred[:,5]-dev['target'][:,5]).mean())
                    checkpoint=directory/f'{step:06d}.pt'
                    torch.save(dict(model=model.state_dict(),step=step,variant=variant,config=cfg,
                        normalization=normal,optimizer=optimizer.state_dict()),checkpoint)
                    checkpoints.append(dict(step=step,primary=score,path=str(checkpoint)))
                    if score<best:
                        best=score;save(directory/'selected.json',checkpoints[-1])
                    model.train()
        save(directory/'evaluations.json',checkpoints)
        save(directory/'complete.json',dict(steps=cfg['steps'],best_primary=best,elapsed_s=time.monotonic()-started))
        del model,optimizer,x,xd;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--stage',choices=['cache','train'],required=True);args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True
    if args.stage=='cache':cache(args.base.resolve())
    else:train_local(args.base.resolve())


if __name__=='__main__':main()
