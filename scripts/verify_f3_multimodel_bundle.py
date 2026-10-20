"""F3-r1 offline identity/strict-load checks; never connects robot hardware."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--device',default='cpu');args=parser.parse_args();root=args.bundle.resolve()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.benchmark=True
    identities=json.loads((root/'model_identity.json').read_text())
    assert set(identities)=={'M0','M1','M2','M3'}
    report={}
    for name in ['M0','M1','M2']:
        entry=identities[name];checkpoint=root/entry['global_path']
        assert hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest()==entry['global_sha256']
        cfg=PreTrainedConfig.from_pretrained(checkpoint);cfg.device=args.device
        cfg.object_conditioning_prototype_path=str(root/'assets/cup_prototype_bank_v1.pt')
        dimension=8 if name=='M0' else 6
        assert cfg.robot_state_feature.shape[0]==dimension
        assert cfg.load_history_steps==(1 if name=='M2' else 0)
        assert not getattr(cfg,'load_feature_gating',False)
        policy=ACTPolicy.from_pretrained(checkpoint,config=cfg).to(args.device).eval()
        batch={'observation.state':torch.zeros((1,dimension),device=args.device),
            'observation.images.top':torch.zeros((1,3,480,640),device=args.device),
            'observation.images.gripper':torch.zeros((1,3,480,640),device=args.device)}
        if name=='M2':batch['observation.load']=torch.zeros((1,1),device=args.device)
        if name=='M0':
            for filename in ['policy_preprocessor.json','policy_postprocessor.json']:
                assert (checkpoint/filename).exists()
        with torch.inference_mode():prediction=policy.predict_action_chunk(batch)
        assert prediction.shape==(1,100,6) and torch.isfinite(prediction).all()
        report[name]=dict(state_dimension=dimension,strict_load=True,finite_synthetic_forward=True)
        del policy,batch,prediction;torch.cuda.empty_cache()
    assert identities['M3']['global_sha256']==identities['M1']['global_sha256']
    assert identities['M3']['global_path']==identities['M1']['global_path']
    assert hashlib.sha256((root/identities['M3']['local_path']).read_bytes()).hexdigest()==identities['M3']['local_sha256']
    print(json.dumps(dict(status='OFFLINE_SMOKE_PASS',variants=report,M3_same_global_as_M1=True,
        M3_local_checks='separate verify_f3_bundle.py',robot_trials='NOT_RUN',common_deployment='PENDING')))


if __name__=='__main__':main()
