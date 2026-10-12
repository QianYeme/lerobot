#!/usr/bin/env python
"""Frozen-Dev counterfactual evaluation for one N0 candidate."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.offline_eval_act_det import collate, parse_episodes
from lerobot.utils.constants import ACTION, OBS_IMAGES

DEV = [3, 5, 12, 15, 21, 28, 33, 39, 41, 47, 53, 58]
MODES = ("C0_normal", "C1_spatial_zero", "C2_spatial_shuffle", "C3_x_transplant", "C4_y_transplant", "C5_full_spatial_transplant")


def h0(x): return x[:, 0] if x.ndim == 3 else x
def move(b, d): return {k: v.to(d) if isinstance(v, torch.Tensor) else v for k, v in b.items()}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--candidate', required=True, choices=['R2_ACTION_LEVEL','R1_RELATIONAL','R3_FILM']); ap.add_argument('--checkpoint',required=True); ap.add_argument('--repo-id',required=True); ap.add_argument('--root',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--episodes',default=','.join(map(str,DEV))); ap.add_argument('--batch-size',type=int,default=4); ap.add_argument('--num-workers',type=int,default=0); a=ap.parse_args()
    episodes=parse_episodes(a.episodes)
    if episodes != DEV: raise ValueError(f'Frozen Dev episodes are {DEV}')
    ckpt=Path(a.checkpoint); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); cfg=PreTrainedConfig.from_pretrained(ckpt); policy=ACTPolicy.from_pretrained(ckpt,config=cfg); policy.eval(); dev=next(policy.parameters()).device
    pre,_=make_pre_post_processors(policy_cfg=cfg,pretrained_path=str(ckpt)); ds=LeRobotDataset(a.repo_id,root=a.root,episodes=episodes,video_backend='pyav'); dl=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,collate_fn=collate,drop_last=False)
    oc=policy.model.object_conditioning; top_idx=policy.model.object_conditioning_top_camera_index; sums={m:{k:0.0 for k in ('all','h0_all','arm','gripper')} for m in MODES}; dim={m:None for m in MODES}; rows=[]; count=0; gen=torch.Generator(device=dev).manual_seed(17)
    with torch.no_grad():
      for raw in dl:
        raw_pad=raw.get('action_is_pad'); b=pre(raw); 
        if 'action_is_pad' not in b and raw_pad is not None: b['action_is_pad']=raw_pad
        b=move(b,dev)
        if 'action_is_pad' not in b: b['action_is_pad']=torch.zeros(b[ACTION].shape[:-1],dtype=torch.bool,device=dev)
        if cfg.image_features: b[OBS_IMAGES]=[b[k] for k in cfg.image_features]
        target=h0(b[ACTION]); pad=b['action_is_pad']; pad=pad[:,0] if pad.ndim==2 else pad; top=b[OBS_IMAGES][top_idx]; base_raw,sim=oc.extract_raw_object_tokens(top); perm=torch.randperm(base_raw.shape[0],generator=gen,device=dev); original_extract=oc.extract_raw_object_tokens; original_spatial=oc.global_spatial_features
        def run(mode):
          if mode=='C0_normal': return h0(policy.model(b)[0])
          donor=base_raw[perm]; base=base_raw[:,0,384:387]; d=donor[:,0,384:387]; chosen=base.clone()
          if mode=='C1_spatial_zero': chosen.zero_()
          elif mode in ('C2_spatial_shuffle','C5_full_spatial_transplant'): chosen=d
          elif mode=='C3_x_transplant': chosen[:,0]=d[:,0]
          elif mode=='C4_y_transplant': chosen[:,1]=d[:,1]
          oc.extract_raw_object_tokens=lambda _: (base_raw,sim); oc.global_spatial_features=lambda _: chosen
          try: return h0(policy.model(b)[0])
          finally: oc.extract_raw_object_tokens=original_extract; oc.global_spatial_features=original_spatial
        preds={m:run(m) for m in MODES}; normal=preds['C0_normal'];
        for m,p in preds.items():
          e=(p-target).abs()*(~pad).unsqueeze(-1); sums[m]['all']+=float(e.mean())*len(target); sums[m]['h0_all']+=float(e.mean())*len(target); sums[m]['arm']+=float(e[:,:-1].mean())*len(target); sums[m]['gripper']+=float(e[:,-1].mean())*len(target); q=e.mean(0).cpu().numpy(); dim[m]=q if dim[m] is None else dim[m]+q*len(target)
        for m in MODES[1:]:
          delta=(preds[m]-normal).cpu().numpy(); rows.append({'mode':m,'h0_delta':delta.tolist(),'mean_abs_arm':float(np.abs(delta[:,:-1]).mean()),'mean_abs_gripper':float(np.abs(delta[:,-1]).mean()),'spatial_delta':(base_raw[perm,0,384:387]-base_raw[:,0,384:387]).cpu().tolist()})
        count+=len(target)
    for m in MODES:
      for k in sums[m]: sums[m][k]/=max(count,1)
      sums[m]['per_action_dim']=(dim[m]/max(count,1)).tolist()
    result={'experiment':'N0_OVERNIGHT_SPATIAL_CONDITIONING_BATCH','candidate':a.candidate,'checkpoint':str(ckpt),'episodes':episodes,'frames':count,'device':str(dev),'metrics':sums,'final_test_used':False,'real_robot_used':False}
    (out/'metrics.json').write_text(json.dumps(result,indent=2)); (out/'counterfactual.json').write_text(json.dumps(rows,indent=2)); (out/'validation.json').write_text(json.dumps({'dev_only':True,'episodes_frozen':episodes==DEV,'candidate':a.candidate,'final_test_used':False,'real_robot_used':False},indent=2)); print(json.dumps({'candidate':a.candidate,'frames':count,'metrics':sums},indent=2))


if __name__=='__main__': main()
