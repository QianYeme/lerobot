"""Frozen F3-r1 inference routing. No robot/camera connections in this module."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.local_load_regulator import LocalLoadRegulator

POSITION_FIELDS = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
                   'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']


class F3Runtime:
    def __init__(self, bundle, model, device='cuda', n_action_steps=1, ensemble=None):
        self.root = Path(bundle).resolve()
        self.model_id, self.device = model, device
        identities = json.loads((self.root/'model_identity.json').read_text())
        self.identity = identities[model]
        assert identities['M1']['global_sha256'] == identities['M3']['global_sha256']
        checkpoint = self.root/self.identity['global_path']
        actual = hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest()
        if actual != self.identity['global_sha256']:
            raise ValueError('BLOCKED_MODEL_IDENTITY')
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.device = device
        cfg.n_action_steps = n_action_steps
        cfg.temporal_ensemble_coeff = ensemble
        cfg.object_conditioning_prototype_path = str(self.root/'assets/cup_prototype_bank_v1.pt')
        expected_dimension = 8 if model == 'M0' else 6
        assert cfg.robot_state_feature.shape[0] == expected_dimension
        assert cfg.load_history_steps == (1 if model == 'M2' else 0)
        self.policy = ACTPolicy.from_pretrained(checkpoint, config=cfg).to(device).eval().requires_grad_(False)
        self.pre = self.post = None
        if model == 'M0':
            from lerobot.policies.factory import make_pre_post_processors
            self.pre, self.post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(checkpoint),
                preprocessor_overrides={'device_processor': {'device': device}})
        self.norm = json.loads((self.root/'normalization.json').read_text())
        self.mean = {k:torch.tensor(self.norm[k]['mean'], device=device, dtype=torch.float32) for k in ['state','action']}
        self.std = {k:torch.tensor(self.norm[k]['std'], device=device, dtype=torch.float32).clamp_min(1e-8) for k in ['state','action']}
        self.image_mean = torch.tensor([.485,.456,.406], device=device).view(1,3,1,1)
        self.image_std = torch.tensor([.229,.224,.225], device=device).view(1,3,1,1)
        self.local = None
        if model == 'M3':
            local = self.root/self.identity['local_path']
            assert hashlib.sha256(local.read_bytes()).hexdigest() == self.identity['local_sha256']
            self.local = LocalLoadRegulator(local, device)
            assert self.local.bound == 4.832699459791184
        self.reset()

    def reset(self):
        self.policy.reset()
        for processor in [self.pre, self.post]:
            if processor is not None:
                processor.reset()

    def prepare(self, observation):
        fields = POSITION_FIELDS + (['gripper.load','gripper.curr'] if self.model_id == 'M0' else [])
        state = torch.tensor([[float(observation[k]) for k in fields]], device=self.device)
        if not torch.isfinite(state).all():
            raise ValueError('Nonfinite state')
        batch = {'observation.state':state}
        for key in ['top','gripper']:
            image = np.asarray(observation[key])
            if image.shape != (480,640,3) or image.dtype != np.uint8:
                raise ValueError('Expected RGB uint8 camera frame480x640, no silent resize')
            # Match dataset/legacy prepare_observation: uint8 -> float /255 on CPU,
            # then transfer. GPU division can differ by one ULP before DINO.
            value = torch.from_numpy(image.copy()).permute(2,0,1).contiguous().float()/255
            batch['observation.images.'+key] = value.unsqueeze(0).to(self.device)
        if self.model_id == 'M0':
            return self.pre(batch)
        batch['observation.state'] = (state-self.mean['state'])/self.std['state']
        for key in ['top','gripper']:
            name = 'observation.images.'+key
            batch[name] = (batch[name]-self.image_mean)/self.image_std
        if self.model_id == 'M2':
            value = (float(observation['gripper.load'])-self.norm['load']['mean'])/self.norm['load']['std']
            if not np.isfinite(value):
                raise ValueError('Nonfinite load')
            batch['observation.load'] = torch.tensor([[value]], device=self.device, dtype=torch.float32)
        return batch

    @torch.inference_mode()
    def step(self, observation, mode='disabled'):
        if mode != 'disabled' and self.model_id != 'M3':
            raise ValueError('Only M3 has shadow/regulated modes')
        batch = self.prepare(observation)
        normalized = self.policy.select_action(batch)
        native = self.post(normalized).to(self.device) if self.post else normalized*self.std['action']+self.mean['action']
        if native.shape != (1,6) or not torch.isfinite(native).all():
            raise ValueError('Nonfinite or invalid action')
        nominal = native.clone()
        telemetry = None
        if self.local is not None and mode != 'disabled':
            load = torch.tensor([float(observation['gripper.load'])], device=self.device)
            pos = torch.tensor([float(observation['gripper.pos'])], device=self.device)
            native, telemetry = self.local.apply(native, load, pos, mode)
        if not torch.equal(native[:,:5], nominal[:,:5]):
            raise RuntimeError('INVALID_LOCAL_ISOLATION')
        return native[0].cpu().numpy(), nominal[0].cpu().numpy(), telemetry
