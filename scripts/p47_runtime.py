"""Deployment runtime shared by P4.7 private model packages."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors.torch import load_file


FIELDS=['shoulder_pan.pos','shoulder_lift.pos','elbow_flex.pos','wrist_flex.pos','wrist_roll.pos','gripper.pos']


class Runtime:
    def __init__(self, root, device='cuda', n_action_steps=None, ensemble=None, local_on=False):
        self.root=Path(root).resolve()
        self.identity=json.loads((self.root/'manifest.json').read_text())
        role=self.identity['role'];self.role=role;self.device=device
        from lerobot.configs.policies import PreTrainedConfig
        if role=='VDP-S1':
            from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        else:
            from lerobot.policies.act.configuration_act import ACTConfig
        cfg=PreTrainedConfig.from_pretrained(self.root)
        cfg.device=device
        torch.hub.set_dir(str(self.root/'torch_cache/hub'))
        if role!='VDP-S1':
            if n_action_steps is not None:cfg.n_action_steps=n_action_steps
            if ensemble is not None:cfg.temporal_ensemble_coeff=ensemble
        elif n_action_steps is not None or ensemble is not None:
            raise ValueError('Diffusion preserves native2 observations/16 horizon/8 execution; no ACT ensemble')
        if getattr(cfg,'object_conditioning_enabled',False):
            cfg.object_conditioning_prototype_path=str(self.root/'assets/formal6_train90_prototypes.pt')
            torch.hub.set_dir(str(self.root/'torch_cache/hub'))
        if role=='VDP-S1':
            from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy as Policy
            from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors as processors
        else:
            from lerobot.policies.act.modeling_act import ACTPolicy as Policy
            from lerobot.policies.act.processor_act import make_act_pre_post_processors as processors
        self.policy=Policy.from_pretrained(self.root,config=cfg).to(device).eval().requires_grad_(False)
        self.stats=torch.load(self.root/'assets/statistics.pt',map_location='cpu',weights_only=True)
        self.pre,self.post=processors(cfg,self.stats)
        self.raw_top=None;self.local=None;self.local_on=local_on
        self.fields=FIELDS+(['gripper.load','gripper.curr'] if role in ['L1','F3','A1'] else [])
        self.identity['state_fields']=self.fields
        if role=='F3' or role=='NG-F-GL':
            def hook(_module,args,kwargs):return (args[0],self.raw_top,*args[2:]),kwargs
            self.policy.model.object_conditioning.register_forward_pre_hook(hook,with_kwargs=True)
        if role.startswith('NG-'):
            self.local=load_file(str(self.root/'local.safetensors'),device=device)
            self.local_config=json.loads((self.root/'local_config.json').read_text())
            self.load_stats=torch.load(self.root/'assets/load_statistics.pt',weights_only=True)
        elif local_on:raise ValueError('This model has no Local regulator')
        self.reset()

    def reset(self):
        self.policy.reset();self.pre.reset();self.post.reset();self.raw_top=None

    def prepare(self, observation):
        state=torch.tensor([[float(observation[key]) for key in self.fields]],dtype=torch.float32)
        raw={'observation.state':state}
        for name in ['top','gripper']:
            image=np.asarray(observation[name]);assert image.shape==(480,640,3) and image.dtype==np.uint8
            raw['observation.images.'+name]=torch.from_numpy(image.copy()).permute(2,0,1).float().div(255)[None]
        self.raw_top=raw['observation.images.top'].to(self.device)
        if self.role.startswith('NG-'):raw['observation.load']=torch.tensor([[float(observation['gripper.load'])]])
        batch=self.pre(raw)
        if self.role.startswith('NG-'):
            mean=self.load_stats['mean'].to(self.device);std=self.load_stats['std'].to(self.device)
            batch['observation.load']=(batch['observation.load']-mean)/std
        return batch

    def correction(self, batch, position, nominal):
        c=self.local_config;n=c['normalization'];w=self.local
        x=torch.stack([batch['observation.load'][0,0],
            torch.as_tensor((position-n['position']['mean'])/n['position']['std'],device=self.device),
            torch.as_tensor((nominal-n['nominal']['mean'])/n['nominal']['std'],device=self.device)]).float()[None]
        x=torch.nn.functional.gelu(torch.nn.functional.linear(x,w['input.weight'],w['input.bias']))
        x=torch.nn.functional.gelu(torch.nn.functional.linear(x,w['hidden.weight'],w['hidden.bias']))
        return float(c['bound']*torch.tanh(torch.nn.functional.linear(x,w['output.weight'],w['output.bias']))[0,0])

    @torch.inference_mode()
    def step(self, observation):
        batch=self.prepare(observation)
        action=self.post(self.policy.select_action(batch))[0].cpu().numpy()
        nominal=action.copy();delta=0.
        if self.local_on:
            delta=self.correction(batch,float(observation['gripper.pos']),float(nominal[5]));action[5]+=delta
        assert np.isfinite(action).all() and action.shape==(6,)
        return action,nominal,dict(clipped_delta=[delta])
