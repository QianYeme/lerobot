"""P4.3 selected-model runtime: legacy/fixed inputs remain explicit and separate."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors

FIELDS = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos',
          'wrist_roll.pos', 'gripper.pos', 'gripper.load', 'gripper.curr']


class P43Runtime:
    def __init__(self, bundle, model, device='cuda', n_action_steps=1, ensemble=.01):
        self.root = Path(bundle).resolve()
        self.model_id, self.device = model, device
        self.identity = json.loads((self.root/'model_identity_manifest.json').read_text())[model]
        checkpoint = self.root/self.identity['pretrained_path']
        with (checkpoint/'model.safetensors').open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != self.identity['model_sha256']:
            raise ValueError('Selected model weight checksum mismatch')
        torch.hub.set_dir(str(self.root/'torch_cache/hub'))
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.device, cfg.n_action_steps, cfg.temporal_ensemble_coeff = device, n_action_steps, ensemble
        if tuple(cfg.robot_state_feature.shape) != (8,) or tuple(cfg.action_feature.shape) != (6,):
            raise ValueError('P4.3 requires state8D/action6D')
        if cfg.load_history_steps or any(getattr(cfg, key, False) for key in (
            'explicit_spatial_conditioning', 'joint_state_spatial_conditioning',
            'action_level_spatial_conditioning', 'relational_state_spatial_conditioning', 'spatial_film_conditioning')):
            raise ValueError('Unexpected branch for selected P4.3 model')
        if cfg.object_conditioning_enabled != (model != 'A1'):
            raise ValueError('ACT/P1 model identity mismatch')
        if cfg.object_conditioning_enabled:
            cfg.object_conditioning_prototype_path = str(self.root/'assets/formal6_train90_prototypes.pt')
        self.policy = ACTPolicy.from_pretrained(checkpoint, config=cfg).to(device).eval().requires_grad_(False)
        stats = torch.load(self.root/'assets/train90_statistics.pt', map_location='cpu', weights_only=False)
        self.pre, self.post = make_act_pre_post_processors(cfg, stats)
        self.local = None
        self._raw_top = None
        self._hook = None
        if model == 'F3':
            # No parameter changes. Equivalent to the isolated fixed ACT raw-Top boundary.
            def fixed_input(_module, args, kwargs):
                if self._raw_top is None or kwargs.get('object_tokens') is not None:
                    raise ValueError('Fixed P1 raw Top boundary not prepared')
                return (args[0], self._raw_top, *args[2:]), kwargs
            self._hook = self.policy.model.object_conditioning.register_forward_pre_hook(fixed_input, with_kwargs=True)
        self.reset()

    def reset(self):
        self.policy.reset()
        self.pre.reset()
        self.post.reset()
        self._raw_top = None

    def prepare(self, observation):
        state = torch.tensor([[float(observation[key]) for key in FIELDS]], dtype=torch.float32)
        if not torch.isfinite(state).all():
            raise ValueError('Nonfinite state/load/current')
        raw = {'observation.state': state}
        for name in ('top', 'gripper'):
            image = np.asarray(observation[name])
            if image.dtype != np.uint8 or image.shape != (480, 640, 3):
                raise ValueError('Expected RGB uint8 480x640 camera frame')
            raw['observation.images.'+name] = torch.from_numpy(image.copy()).permute(2,0,1).float().div(255)[None]
        self._raw_top = raw['observation.images.top'].to(self.device)
        return self.pre(raw)

    @torch.inference_mode()
    def chunk(self, observation, native=False):
        prediction = self.policy.predict_action_chunk(self.prepare(observation))
        return self.post(prediction) if native else prediction

    @torch.inference_mode()
    def step(self, observation):
        normalized = self.policy.select_action(self.prepare(observation))
        native = self.post(normalized)
        if native.shape != (1, 6) or not torch.isfinite(native).all():
            raise ValueError('Invalid predicted action')
        value = native[0].detach().cpu().numpy()
        return value, value.copy(), None
