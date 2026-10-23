"""R5 paired Train36/Full60 deployment, retaining each model's normalizer."""
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.f5_runtime import F5Local, F5Runtime
from lerobot.utils.r5_protocol import BOUND, verify_bundle


class R5Runtime(F5Runtime):
    def __init__(self,bundle,model,device='cuda',n_action_steps=1,ensemble=.01):
        self.root=Path(bundle).resolve();self.model_name=model;self.model_id='M2';self.device=device
        self.identity=verify_bundle(self.root,model)
        checkpoint=self.root/self.identity['global_path'];cfg=PreTrainedConfig.from_pretrained(checkpoint)
        cfg.device=device;cfg.n_action_steps=n_action_steps;cfg.temporal_ensemble_coeff=ensemble
        cfg.object_conditioning_prototype_path=str(self.root/'assets/cup_prototype_bank_v1.pt')
        if (cfg.robot_state_feature.shape[0],cfg.action_feature.shape[0],cfg.load_history_steps)!=(6,6,1):
            raise ValueError('R5 requires 6D state / 6D action / single-frame global load')
        self.policy=ACTPolicy.from_pretrained(checkpoint,config=cfg).to(device).eval().requires_grad_(False)
        self.pre=self.post=None
        self.norm=json.loads((self.root/self.identity['normalization_path']).read_text())
        self.mean={k:torch.tensor(self.norm[k]['mean'],device=device,dtype=torch.float32) for k in ['state','action']}
        self.std={k:torch.tensor(self.norm[k]['std'],device=device,dtype=torch.float32).clamp_min(1e-8) for k in ['state','action']}
        self.image_mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1)
        self.image_std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)
        self.local=F5Local(self.root/self.identity['local_path'],device) if self.identity.get('local_path') else None
        if self.local is not None:
            saved=torch.load(self.root/self.identity['local_path'],map_location='cpu',weights_only=True)
            if saved.get('bound',BOUND)!=BOUND or self.identity['bound']!=BOUND:raise ValueError('Residual bound mismatch')
            if 'global_sha256' in saved and saved['global_sha256']!=self.identity['global_sha256']:
                raise ValueError('Local attached to wrong backbone')
            # Legacy F4.2 lacks backbone/bound metadata; its exact pair is frozen in verify_bundle.
            for key in ['mean','std']:
                if self.local.normalization['load'][key]!=self.norm['load'][key]:raise ValueError('Load normalization mismatch')
        self.reset()

    def prepare(self,observation):
        batch=super().prepare(observation)
        # Match R4/Full60's cached float32 subtract/divide, not Python float64 rounding.
        # A one-ULP load difference caused >0.001 native output error on a real RGB fixture.
        normal=self.norm['load']
        value=(np.float32(observation['gripper.load'])-np.float32(normal['mean']))/np.float32(normal['std'])
        if not np.isfinite(value):raise ValueError('Nonfinite normalized load')
        batch['observation.load']=torch.tensor([[value]],device=self.device,dtype=torch.float32)
        return batch
