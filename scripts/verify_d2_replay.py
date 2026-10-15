"""Validate cached D2 replay against actual code and legal episode targets."""
import json
import logging
from pathlib import Path
import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
from lerobot.robots.utils import ensure_safe_goal_position

logging.disable(logging.CRITICAL)
root=Path('/root/autodl-tmp/lerobot/results/experiments/D2_P1_ACTION_CHAIN_DIAGNOSIS/P1')
report={}
for p in sorted(root.glob('episode_*.npz')):
    d=np.load(p)
    chunks=d['chunks'];state=d['state'];target=d['target'];frames=d['frames']
    assert np.all(np.diff(frames)==1)
    ens=ACTTemporalEnsembler(0.01,chunks.shape[1])
    errors=[]
    for t,chunk in enumerate(chunks):
        actual=ens.update(torch.from_numpy(chunk)[None])[0].numpy()
        start=max(0,t-chunks.shape[1]+1)
        prior=np.stack([chunks[s,t-s] for s in range(start,t+1)])
        weights=np.exp(-0.01*np.arange(len(prior)))
        reference=(prior*weights[:,None]).sum(0)/weights.sum()
        errors.append(float(np.max(np.abs(actual-reference))))
    clamp_error=0
    for limit in [2.0,5.0]:
        for i in range(0,len(state),50):
            pairs={str(k):(float(chunks[i,0,k]),float(state[i,k])) for k in range(6)}
            expected=ensure_safe_goal_position(pairs,limit)
            replay=state[i,:6]+np.clip(chunks[i,0]-state[i,:6],-limit,limit)
            clamp_error=max(clamp_error,max(abs(expected[str(k)]-replay[k]) for k in range(6)))
    # Legal same-episode future targets and mask: no repeated tail labels.
    err=np.zeros(6);count=0
    for i in range(len(target)):
        valid=min(chunks.shape[1],len(target)-i)
        err+=np.abs(chunks[i,:valid]-target[i:i+valid]).sum(0)
        count+=valid
    report[p.stem]={'temporal_max_abs_error':max(errors),'clamp_max_abs_error':float(clamp_error),
                   'full_chunk_valid_targets':count,'full_chunk_per_joint_mae':(err/count).tolist(),
                   'tail_mask':'future index < episode length',
                   'phase':'unknown'}
assert max(x['temporal_max_abs_error'] for x in report.values())<1e-3
assert max(x['clamp_max_abs_error'] for x in report.values())<1e-4
(root/'replay_validation.json').write_text(json.dumps(report,indent=2))
print('PASS actual temporal weights, actual clamp helper, same-episode full-chunk masks',len(report))
