"""Four F5 inference routes, sharing established RGB/state preprocessing only."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.f3_runtime import F3Runtime
from lerobot.policies.act.local_load_regulator import LocalLoadRegulator
from lerobot.utils.f5_protocol import BOUND, verify_bundle


class F5Local(LocalLoadRegulator):
    def __init__(self,path,device):
        # New checkpoints intentionally have no legacy C1-load variant tag.
        nn.Module.__init__(self)
        saved=torch.load(path,map_location='cpu',weights_only=True)
        self.bound=BOUND;self.normalization=saved['normalization']
        for field in ['load','position','nominal']:
            n=self.normalization[field]
            if not np.isfinite([n['mean'],n['std']]).all() or n['std']<=0:raise ValueError('Invalid local normalization')
        self.input=nn.Linear(3,32);self.hidden=nn.Linear(32,32);self.output=nn.Linear(32,1)
        self.load_state_dict(saved['model'],strict=True);self.to(device).eval().requires_grad_(False)


class F5Runtime(F3Runtime):
    def __init__(self,bundle,model,device='cuda',n_action_steps=1,ensemble=.01):
        self.root=Path(bundle).resolve();self.model_name=model;self.device=device
        self.identity=verify_bundle(self.root,model)
        # Reuse the already-verified F3 RGB/state preparation, not its gates or model selection.
        self.model_id={'P1-Hist':'M0','Strong-G':'M2','L-Load':'M1','New-GL':'M2'}[model]
        checkpoint=self.root/self.identity['global_path'];cfg=PreTrainedConfig.from_pretrained(checkpoint)
        cfg.device=device;cfg.n_action_steps=n_action_steps;cfg.temporal_ensemble_coeff=ensemble
        cfg.object_conditioning_prototype_path=str(self.root/'assets/cup_prototype_bank_v1.pt')
        if cfg.robot_state_feature.shape[0]!=(8 if model=='P1-Hist' else 6):raise ValueError('State dimension mismatch')
        if cfg.load_history_steps!=(1 if model in ['Strong-G','New-GL'] else 0):raise ValueError('Global load route mismatch')
        if cfg.action_feature.shape[0]!=6:raise ValueError('Expected six actions')
        self.policy=ACTPolicy.from_pretrained(checkpoint,config=cfg).to(device).eval().requires_grad_(False)
        self.pre=self.post=None
        if model=='P1-Hist':
            from lerobot.policies.factory import make_pre_post_processors
            self.pre,self.post=make_pre_post_processors(policy_cfg=cfg,pretrained_path=str(checkpoint),preprocessor_overrides={'device_processor':{'device':device}})
        self.norm=json.loads((self.root/'normalization.json').read_text())
        self.mean={k:torch.tensor(self.norm[k]['mean'],device=device,dtype=torch.float32) for k in ['state','action']}
        self.std={k:torch.tensor(self.norm[k]['std'],device=device,dtype=torch.float32).clamp_min(1e-8) for k in ['state','action']}
        self.image_mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1)
        self.image_std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)
        self.local=F5Local(self.root/self.identity['local_path'],device) if model in ['L-Load','New-GL'] else None
        if self.local is not None:
            for k in ['mean','std']:
                if self.local.normalization['load'][k]!=self.norm['load'][k]:raise ValueError('Local/global load normalizer mismatch')
        self.reset()

    @torch.inference_mode()
    def step(self,observation):
        batch=self.prepare(observation)
        normalized=self.policy.select_action(batch)
        native=self.post(normalized).to(self.device) if self.post else normalized*self.std['action']+self.mean['action']
        if native.shape!=(1,6) or not torch.isfinite(native).all():raise ValueError('Invalid native action')
        nominal=native.clone();telemetry=None
        if self.local is not None:
            load=torch.tensor([float(observation['gripper.load'])],device=self.device)
            pos=torch.tensor([float(observation['gripper.pos'])],device=self.device)
            # Exactly once: AFTER global temporal ensemble and denormalization,
            # BEFORE unchanged robot relative-target limiting.
            native,telemetry=self.local.apply(native,load,pos,'regulated')
        if not torch.equal(native[:,:5],nominal[:,:5]):raise RuntimeError('INVALID_LOCAL_ISOLATION')
        return native[0].cpu().numpy(),nominal[0].cpu().numpy(),telemetry
