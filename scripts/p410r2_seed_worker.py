"""Seed2 known-start replication, exact r5 recipe except authorized seed/gate."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path('/root/autodl-tmp/lerobot-main')
R5=ROOT/'experiments/p4_public_benchmark/p4_8_libero_pilot/index5'
WORK=ROOT/'experiments/p4_public_benchmark/p4_10_gate_sweep/index3'
sys.path.insert(0,str(R5/'workspace/scripts'))
import p48_r5_common as common
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--variant',choices=['G0','G1e-2','G5e-2','G1e-1','NOOBJ'],required=True)
p.add_argument('--mode',choices=['smoke','bench','train'],required=True);p.add_argument('--tag',default='single');args=p.parse_args()
values={'G0':0.,'G1e-2':.01,'G5e-2':.05,'G1e-1':.1}
folder=WORK/'result/known_seed2'/args.variant;folder.mkdir(parents=True,exist_ok=True)
for name in ['prototype_gate.json','prototype_bank.pt','dataset_split_manifest.json','train35_statistics.pt']:
    dest=folder/name
    if not dest.exists():dest.symlink_to(R5/'result'/name)
original_seed=common.seed
common.seed=lambda ignored=1:original_seed(2)
original_factory=common.make_policy
def factory(kind):
    cfg,policy=original_factory(kind)
    if kind!='act':
        import torch
        with torch.no_grad():policy.model.object_conditioning.gate.fill_(values[args.variant])
    return cfg,policy
common.make_policy=factory
def order(length,epoch):
    indices=np.random.default_rng(np.random.SeedSequence([2,epoch])).permutation(length)
    return [indices[i:i+8] for i in range(0,len(indices),8)]
common.epoch_order=order
source=(R5/'workspace/scripts/p48_r5_worker.py').read_text().replace('10000','5000')
namespace={'__name__':'p410r2_seed2_replication','__file__':str(R5/'workspace/scripts/p48_r5_worker.py')}
exec(compile(source,'frozen_r5_worker_budget5000','exec'),namespace)
namespace['OUT']=folder
original_effective=namespace['effective']
def effective(kind,cfg,policy,preset):
    original_effective(kind,cfg,policy,preset)
    path=folder/'effective_configs'/f'{kind}.yaml';record=json.loads(path.read_text())
    record.update(seed=2,steps=5000,classification='REPLICATION_ON_KNOWN_STARTS',
        sampler='numpy PCG64 SeedSequence([2,epoch]); exact epoch/cursor checkpointed',
        actual_gate_initial=None if kind=='act' else values[args.variant],
        wrapper_sha256=common.sha(Path(__file__)))
    if kind!='act':record['object_conditioning']['gate']=f'learnable signed scalar, init{values[args.variant]}'
    common.write(path,record)
namespace['effective']=effective
kind='act' if args.variant=='NOOBJ' else 'fixed'
sys.argv=[sys.argv[0],'--kind',kind,'--mode',args.mode,'--tag',args.tag]
namespace['main']()
