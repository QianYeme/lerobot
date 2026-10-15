"""B3 real module/forward identity and P1 cached normalized replay equivalence."""
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy,ACTTemporalEnsembler
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.offline_eval_act_det import collate

root=Path('/root/autodl-tmp/lerobot')
out=root/'results/experiments/D2_1_ACTION_CHAIN_EVIDENCE_COMPLETION'
out.mkdir(parents=True,exist_ok=True)
(out/'remote_dirty.patch').write_text(subprocess.check_output(['git','diff'],cwd=root,text=True))
paths={'B3':root/'results/experiments/T3_JOINT_STATE_SPATIAL_CONDITIONING/T3_B3_s1/checkpoints/100000/pretrained_model',
       'P1':root/'results/experiments/T0_POSITION_DISCOVERY_ACT_VS_OCACT/P1_ocfix2_s1/checkpoints/100000/pretrained_model',
       'R3':root/'results/experiments/N0_OVERNIGHT_SPATIAL_CONDITIONING_BATCH/R3_FILM_s1/checkpoints/100000/pretrained_model'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
provenance={'remote_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            'torch':torch.__version__,'assets':{}}
for label,p in paths.items():
    config=json.loads((p/'config.json').read_text())
    provenance['assets'][label]={'files':{q.name:sha(q) for q in p.iterdir() if q.is_file()},
        'prototype_path':config.get('object_conditioning_prototype_path')}
    prototype=config.get('object_conditioning_prototype_path')
    if prototype and Path(prototype).exists():provenance['assets'][label]['prototype_sha256']=sha(prototype)
cfg=PreTrainedConfig.from_pretrained(paths['B3']);cfg.device='cuda'
policy=ACTPolicy.from_pretrained(paths['B3'],config=cfg).to('cuda').eval()
pre,_=make_pre_post_processors(policy_cfg=cfg,pretrained_path=str(paths['B3']))
ds=LeRobotDataset('formal4',root=root/'formal4_data/kind_merged_nomaster',episodes=[3],video_backend='pyav')
batch=pre(collate([ds[0]]));batch={k:v.to('cuda') if isinstance(v,torch.Tensor) else v for k,v in batch.items()};batch.pop('action',None)
oc=getattr(policy.model,'object_conditioning',None);calls=[]
handle=oc.register_forward_hook(lambda module,args,output:calls.append(True)) if oc else None
with torch.no_grad():policy.predict_action_chunk(batch)
if handle:handle.remove()
identity={'config_enabled':cfg.object_conditioning_enabled,'module_present':oc is not None,
          'forward_path_enabled':bool(calls),'status':'OCACT_VERIFIED' if oc is not None and calls else 'UNRESOLVED',
          'training_provenance':'T3 manifest preserves OC-ACT branch; historical per-step invocation unverified',
          'pure_act_baseline':'MISSING','checkpoint_identity_not_training_effect_claim':True}
with safe_open(str(paths['B3']/'model.safetensors'),framework='pt',device='cpu') as tensors:
    identity['object_checkpoint_keys']=[k for k in tensors.keys() if 'object_conditioning' in k]
identity['module_summary']=str(oc)
for filename in ['config.json','train_config.json']:
    (out/('B3_'+filename)).write_bytes((paths['B3']/filename).read_bytes())
(out/'b3_identity_audit.json').write_text(json.dumps(identity,indent=2))
del policy;torch.cuda.empty_cache()
cfg=PreTrainedConfig.from_pretrained(paths['P1']);cfg.device='cuda'
_,post=make_pre_post_processors(policy_cfg=cfg,pretrained_path=str(paths['P1']))
checks=[]
for p in sorted((root/'results/experiments/D2_P1_ACTION_CHAIN_DIAGNOSIS/P1').glob('episode_*.npz')):
    d=np.load(p);normal=d['normalized_chunks'];physical=d['chunks'];frames=d['frames']
    assert len(frames)==479 and np.all(np.diff(frames)==1)
    en=ACTTemporalEnsembler(.01,normal.shape[1]);ep=ACTTemporalEnsembler(.01,normal.shape[1]);errors=[]
    for i in range(len(frames)):
        n=en.update(torch.from_numpy(normal[i])[None].to('cuda'))
        a=post(n).cpu().numpy()[0]
        b=ep.update(torch.from_numpy(physical[i])[None])[0].numpy()
        errors.append(float(np.max(np.abs(a-b))))
    checks.append({'episode_file':p.name,'max_abs_error':max(errors),'frames':len(frames),'shape':list(normal.shape),'cache_sha256':sha(p)})
assert max(x['max_abs_error'] for x in checks)<.001
(out/'replay_equivalence_validation.json').write_text(json.dumps({'normalized_then_unnormalized_vs_physical_ensemble':'PASS','tolerance':.001,'checks':checks},indent=2))
(out/'provenance.json').write_text(json.dumps(provenance,indent=2))
cache_root=root/'results/experiments/D2_P1_ACTION_CHAIN_DIAGNOSIS/P1'
horizons={}
for p in sorted(cache_root.glob('episode_*.npz')):
    d=np.load(p);target=d['target'];chunks=d['chunks'];z={}
    for horizon in [1,5,10,30,100]:
        error=np.zeros(6);count=0
        for i in range(len(target)):
            valid=min(horizon,chunks.shape[1],len(target)-i)
            error+=np.abs(chunks[i,:valid]-target[i:i+valid]).sum(0);count+=valid
        z[str(horizon)]={'per_joint_mae':(error/count).tolist(),'valid_target_count':count}
    horizons[p.stem]=z
(out/'horizon_metrics.json').write_text(json.dumps(horizons,indent=2))
print(identity,flush=True);print('P1 equivalence PASS',max(x['max_abs_error'] for x in checks),flush=True)
