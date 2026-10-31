"""Shared frozen data/model interface for the three r5 pilot jobs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path('/root/autodl-tmp/lerobot-main')
WORK=ROOT/'experiments/p4_public_benchmark/p4_8_libero_pilot/index5'
OUT=WORK/'result'
OLD=WORK.parent/'index1/result'
DATA=Path('/root/autodl-tmp/p48_data/lerobot_libero')
sys.path.insert(0,str(OLD/'workspace/lerobot/src'))

import numpy as np
import torch
from lerobot.configs.types import FeatureType,PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.partial');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def seed(value=1):
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)


def rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore_rng(value):
    random.setstate(value['python']);np.random.set_state(value['numpy'])
    torch.set_rng_state(value['cpu']);torch.cuda.set_rng_state_all(value['cuda'])


def config():
    return ACTConfig(input_features={'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(8,)),
        **{key:PolicyFeature(type=FeatureType.VISUAL,shape=(3,256,256)) for key in ['observation.images.image','observation.images.image2']}},
        output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(7,))})


def object_class():
    path=WORK/'workspace/object_conditioning.py'
    expected=json.loads((OUT/'prototype_numeric_validation.json').read_text())
    assert sha(path)==expected['dino_source_sha256']
    assert sha(Path(torch.hub.get_dir())/'checkpoints/dinov2_vits14_pretrain.pth')==expected['dino_weights_sha256']
    spec=importlib.util.spec_from_file_location('p48_r5_original_object_module',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.ObjectConditioning


class P1Policy(ACTPolicy):
    """Stock ACT plus the original OC module at the original pre-encoder fusion location."""
    def __init__(self,cfg,kind):
        super().__init__(cfg)
        self.kind=kind
        oc=SimpleNamespace(object_conditioning_dino_model='dinov2_vits14',
            object_conditioning_prototype_path=str(OUT/'prototype_bank.pt'),object_conditioning_num_prototypes=4,
            object_conditioning_num_object_tokens=5,object_conditioning_adapter_dim=256,
            object_conditioning_temperature=.20,object_conditioning_gate_init=0.,dim_model=512,n_heads=8)
        self.model.object_conditioning=object_class()(oc)
        self.model.encoder.register_forward_pre_hook(self._object_fusion,with_kwargs=True)
        self.raw_top=None

    def _set_top(self,batch):
        self.raw_top=batch['observation.images.image'] if self.kind=='legacy' else batch['_oc_raw_top']

    def _object_fusion(self,module,args,kwargs):
        assert self.raw_top is not None
        tokens=self.model.object_conditioning(args[0],self.raw_top)
        return (tokens,)+args[1:],kwargs

    def forward(self,batch):
        self._set_top(batch)
        return super().forward(batch)

    def predict_action_chunk(self,batch):
        self._set_top(batch)
        return super().predict_action_chunk(batch)


def make_policy(kind):
    assert kind in ['act','legacy','fixed']
    seed(1);cfg=config()
    if kind!='act':assert sha(OUT/'prototype_bank.pt')==json.loads((OUT/'prototype_gate.json').read_text())['bank_sha256']
    policy=ACTPolicy(cfg) if kind=='act' else P1Policy(cfg,kind)
    policy=policy.cuda()
    if kind!='act':
        assert not any(p.requires_grad for p in policy.model.object_conditioning.dino.parameters())
    return cfg,policy


def model_hash(policy,common_only=False):
    digest=hashlib.sha256()
    for name,value in policy.state_dict().items():
        if common_only and name.startswith('model.object_conditioning.'):continue
        array=value.detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()


class CachedData:
    def __init__(self,split='train'):
        folder=OUT/'data_cache'/split
        self.images=np.load(folder/'images.npy',mmap_mode='r')
        self.arrays=dict(np.load(folder/'arrays.npz'))
        self.stats=torch.load(OUT/'train35_statistics.pt',weights_only=True)
        self.pre,_=make_act_pre_post_processors(config(),self.stats)

    def __len__(self):return len(self.images)

    def batch(self,indices):
        indices=np.asarray(indices,dtype=np.int64)
        end=self.arrays['last_index'][indices,None]
        future=indices[:,None]+np.arange(100)[None]
        raw={'observation.state':torch.from_numpy(self.arrays['state'][indices].copy()),
             'action':torch.from_numpy(self.arrays['action'][np.minimum(future,end)].copy()),
             'action_is_pad':torch.from_numpy(future>end)}
        for i,key in enumerate(['observation.images.image','observation.images.image2']):
            raw[key]=torch.from_numpy(np.array(self.images[indices,i],copy=True)).cuda().float()/255
        top=raw['observation.images.image'].clone()
        result=self.pre(raw);result['_oc_raw_top']=top
        return result


def update(policy,opt,batch,clip):
    policy.train();opt.zero_grad(set_to_none=True)
    loss,parts=policy(batch);assert torch.isfinite(loss),'Non-finite loss'
    loss.backward();grad=torch.nn.utils.clip_grad_norm_(policy.parameters(),clip,error_if_nonfinite=True)
    opt.step();return float(loss.detach()),float(grad),parts


def epoch_order(length,epoch):
    order=np.random.default_rng(np.random.SeedSequence([1,epoch])).permutation(length)
    return [order[i:i+8] for i in range(0,len(order),8)]
