"""R5.11 historical133k route with the validated R5 preprocessing."""
import json
from pathlib import Path
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.r5_runtime import R5Runtime
from lerobot.utils.r5_11_protocol import verify_bundle


class R511Runtime(R5Runtime):
    def __init__(self,bundle,model,device='cuda',n_action_steps=1,ensemble=.01):
        self.root=Path(bundle).resolve();self.model_name=model;self.model_id='M2';self.device=device
        self.identity=verify_bundle(self.root,model)
        checkpoint=self.root/self.identity['global_path'];cfg=PreTrainedConfig.from_pretrained(checkpoint)
        cfg.device=device;cfg.n_action_steps=n_action_steps;cfg.temporal_ensemble_coeff=ensemble
        cfg.object_conditioning_prototype_path=str(self.root/'assets/cup_prototype_bank_v1.pt')
        if (cfg.robot_state_feature.shape[0],cfg.action_feature.shape[0],cfg.load_history_steps)!=(6,6,1):raise ValueError('R5.11 requires6D state/action and independent single-frame load')
        self.policy=ACTPolicy.from_pretrained(checkpoint,config=cfg).to(device).eval().requires_grad_(False)
        self.pre=self.post=self.local=None
        self.norm=json.loads((self.root/self.identity['normalization_path']).read_text())
        self.mean={k:torch.tensor(self.norm[k]['mean'],device=device,dtype=torch.float32) for k in ['state','action']}
        self.std={k:torch.tensor(self.norm[k]['std'],device=device,dtype=torch.float32).clamp_min(1e-8) for k in ['state','action']}
        self.image_mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1)
        self.image_std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)
        self.reset()
