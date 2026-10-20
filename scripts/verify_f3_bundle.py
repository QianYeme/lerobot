"""Offline bundle verification only: never opens serial/cameras or executes actions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.local_load_regulator import LocalLoadRegulator


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--device',default='cpu');args=parser.parse_args();root=args.bundle.resolve()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.benchmark=True
    manifest=json.loads((root/'manifest.json').read_text())
    for name,expected in manifest['sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
    local=LocalLoadRegulator(root/'local/010000.pt',args.device)
    with np.load(root/'validation/local_fixture.npz') as z:
        action=torch.tensor(z['nominal_native'],device=args.device,dtype=torch.float32)
        load=torch.tensor(z['raw_load'],device=args.device,dtype=torch.float32)
        pos=torch.tensor(z['position'],device=args.device,dtype=torch.float32)
        disabled,log=local.apply(action,mode='disabled');assert disabled is action and log is None
        shadow,log=local.apply(action,load,pos,mode='shadow');assert shadow is action
        final,_=local.apply(action,load,pos,mode='regulated')
        assert torch.equal(final[:,:5],action[:,:5])
        error=float(np.abs(log['predicted_delta'].cpu().numpy()-z['expected_delta']).max())
        assert error<1e-4, error
        assert log['clipped_delta'].abs().max()<=local.bound
    cfg=PreTrainedConfig.from_pretrained(root/'global')
    cfg.device=args.device
    cfg.object_conditioning_prototype_path=str(root/'assets/cup_prototype_bank_v1.pt')
    assert cfg.robot_state_feature.shape[0]==6 and cfg.load_history_steps==0
    model=ACTPolicy.from_pretrained(root/'global',config=cfg).to(args.device).eval()
    batch={'observation.state':torch.zeros((1,6),device=args.device)}
    for key in ['observation.images.top','observation.images.gripper']:
        batch[key]=torch.zeros((1,3,480,640),device=args.device)
    with torch.inference_mode():prediction=model.predict_action_chunk(batch)
    assert prediction.shape==(1,100,6) and torch.isfinite(prediction).all()
    print(json.dumps(dict(status='OFFLINE_SMOKE_PASS',max_local_fixture_error=error,
        global_strict_load=True,disable_shadow_arm_bound_checks=True,
        deployment_hardware_stage0='NOT_RUN',robot_trials='NOT_RUN')))


if __name__=='__main__':main()
