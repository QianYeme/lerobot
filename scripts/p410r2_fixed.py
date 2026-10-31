"""Authorized A2: fixed alpha=.05, excluded from the optimizer, r0 5k recipe."""
from pathlib import Path
import sys
import importlib.util
import json

ROOT=Path('/root/autodl-tmp/lerobot-main')
OLD=ROOT/'experiments/p4_public_benchmark/p4_10_gate_sweep/index1/workspace/scripts/p410_worker.py'
WORK=ROOT/'experiments/p4_public_benchmark/p4_10_gate_sweep/index3'
sys.path.insert(0,str(OLD.parent))
source=OLD.read_text().replace('float(policy.model.object_conditioning.gate.grad)',
    'float(policy.model.object_conditioning.gate.grad) if policy.model.object_conditioning.gate.grad is not None else 0.')
namespace={'__name__':'p410r2_disposable_fixed_worker','__file__':str(OLD)}
exec(compile(source,str(OLD),'exec'),namespace)
namespace['OUT']=WORK/'result/fixed_alpha';namespace['VALUES']={'GFIXED':.05}
original=namespace['old'].make_policy

def factory(kind):
    cfg,policy=original(kind)
    gate=policy.model.object_conditioning.gate
    gate.requires_grad_(False)
    params=policy.get_optim_params
    def filtered():
        value=params()
        if isinstance(value,list) and value and isinstance(value[0],dict):
            result=[dict(group,params=[p for p in group['params'] if p is not gate]) for group in value]
            assert all(p is not gate for group in result for p in group['params'])
        else:
            result=[p for p in value if p is not gate];assert all(p is not gate for p in result)
        namespace['old'].write(WORK/'result/fixed_gate_optimizer_exclusion.json',dict(
            gate_requires_grad=False,gate_in_optimizer=False,alpha=.05,seed=1,
            parameter_filter_applied_before_optimizer_build=True,
            scope='fixed alpha mechanism control; not just a learnable initialization change',
            source_worker_sha256=namespace['old'].sha(OLD),wrapper_sha256=namespace['old'].sha(Path(__file__))))
        return result
    policy.get_optim_params=filtered
    return cfg,policy

namespace['old'].make_policy=factory
namespace['main']()
