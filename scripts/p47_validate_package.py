"""Real Dev15 streaming export equivalence; no hardware/Final15 access."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);args=p.parse_args()
    root=args.bundle.resolve();sys.path.insert(0,str(root/'source/src'));sys.path.insert(0,str(root))
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from p47_runtime import Runtime
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    project=Path('/root/autodl-tmp/lerobot-main')
    split=project/'experiments/p4_final_p1_training/p4_0_1_training_readiness_unblock/index1/result/dev15_episode_ids.json'
    ids=json.loads(split.read_text());dataset=LeRobotDataset('QYyyyyyyy/formal6_merged',root=project/'outputs/p4_0/formal6_merged',episodes=[ids[0]],video_backend='torchcodec')
    identity=json.loads((root/'manifest.json').read_text());role=identity['role']
    frames=list(range(9)) if role=='VDP-S1' else [0,1,2]
    names=dataset.meta.features['observation.state']['names'];observations=[]
    for index in frames:
        raw=dataset[index];obs=dict(zip(names,raw['observation.state'].tolist()))
        for camera in ['top','gripper']:
            obs[camera]=(raw['observation.images.'+camera].permute(1,2,0).numpy()*255).round().astype(np.uint8)
        observations.append(obs)
    runtime=Runtime(root,n_action_steps=None if role=='VDP-S1' else 1,ensemble=None if role=='VDP-S1' else .01)
    def collect():
        runtime.reset();torch.manual_seed(4701)
        return np.stack([runtime.step(obs)[0] for obs in observations])
    exported=collect();assert np.isfinite(exported).all()
    source=torch.load(identity['source_checkpoint_path'],map_location='cpu',weights_only=False)
    runtime.policy.load_state_dict(source['model'],strict=True);del source
    original=collect();difference=float(np.abs(exported-original).max());assert difference<=1e-5
    reset=collect();assert np.max(np.abs(reset-original))<=1e-5
    local_difference=None
    if role.startswith('NG-'):
        local=torch.load(identity['local_source_path'],map_location='cpu',weights_only=False)
        assert all(torch.equal(value.cpu(),local['model'][key]) for key,value in runtime.local.items())
        runtime.local_on=True;on=collect();assert np.array_equal(original[:,:5],on[:,:5]);assert np.max(np.abs(on[:,5]-original[:,5]))<=runtime.local_config['bound']+1e-5
        local_difference=float(np.abs(on[:,5]-original[:,5]).mean())
    batch=runtime.prepare(observations[0])
    mean=runtime.stats['observation.state']['mean'].to(runtime.device);std=runtime.stats['observation.state']['std'].to(runtime.device)
    raw=torch.tensor([[observations[0][name] for name in runtime.fields]],device=runtime.device)
    if role=='VDP-S1':
        low=runtime.stats['observation.state']['min'].to(runtime.device);high=runtime.stats['observation.state']['max'].to(runtime.device)
        expected=(raw-low)/(high-low).clamp_min(1e-8)*2-1
    else:expected=(raw-mean)/std
    assert torch.allclose(expected,batch['observation.state'],atol=1e-5)
    result=dict(status='PASS',role=role,dev_episode=ids[0],frames=frames,max_export_source_native_difference=difference,
                reset_max_difference=float(np.abs(reset-original).max()),local_arm_unchanged=role.startswith('NG-'),
                local_mean_abs_native=local_difference,normalization_check='PASS',robot_operated=False,final15_accessed=False,
                scope='Source-checkpoint versus safe exported weights through same reviewed streaming runtime; not closed-loop evidence')
    (root/'runtime_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
