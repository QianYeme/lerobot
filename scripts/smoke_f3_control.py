"""Server real-Dev observation checks for unified F3 inference; NO robot calls."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.act.f3_runtime import F3Runtime, POSITION_FIELDS


def main():
    sys.path.insert(0,'/root/autodl-tmp/F2_1_r1_20261019/source/scripts')
    from run_f2_1 import Experiment
    bundle=Path('/root/autodl-tmp/F3_r1_transfer_20261020/bundle')
    original=Path('/root/autodl-tmp/F2_1_r1_20261019')
    exp=Experiment(original/'inputs',Path('/root/autodl-tmp/lerobot/formal4_data/kind_merged_nomaster'),original/'results')
    raw=next(iter(exp.loader('dev',workers=0)));raw={k:v[:1] for k,v in raw.items()}
    state=raw['observation.state'][0].numpy()
    obs={field:float(state[i]) for i,field in enumerate(POSITION_FIELDS+['gripper.load','gripper.curr'])}
    for key in ['top','gripper']:
        obs[key]=np.rint(raw['observation.images.'+key][0].permute(1,2,0).numpy()*255).astype(np.uint8)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.benchmark=True
    results={};saved_nominal=None
    for model in ['M0','M1','M2','M3']:
        runtime=F3Runtime(bundle,model)
        if model=='M0':
            reference=runtime.pre(raw);reference.pop('action',None)
        else:
            reference=exp.batch(raw,'A1' if model=='M2' else 'A0');reference.pop('action')
        actual=runtime.prepare(obs)
        feature_errors={k:float((actual[k]-reference[k]).abs().max()) for k in actual if isinstance(actual[k],torch.Tensor) and k in reference}
        assert max(feature_errors.values())<2e-6,feature_errors
        with torch.inference_mode():
            normalized=runtime.policy.predict_action_chunk(reference)[:,0]
            native=runtime.post(normalized).cpu().numpy()[0] if runtime.post else (normalized*runtime.std['action']+runtime.mean['action']).cpu().numpy()[0]
        runtime.reset();action,nominal,_=runtime.step(obs)
        error=float(np.abs(action-native).max());assert error<1e-4,error
        if model=='M1':saved_nominal=nominal.copy()
        if model=='M3':
            assert np.array_equal(saved_nominal,nominal)
            for mode in ['shadow','regulated']:
                runtime.reset();action,nominal,telemetry=runtime.step(obs,mode)
                assert np.array_equal(action[:5],nominal[:5])
                assert telemetry['clipped_delta'].abs().max()<=runtime.local.bound
                if mode=='shadow':assert np.array_equal(action,nominal)
            no_load={k:v for k,v in obs.items() if k not in ['gripper.load','gripper.curr']}
            runtime.reset();disabled,_,_=runtime.step(no_load,'disabled')
            assert np.array_equal(disabled,saved_nominal)
        results[model]=dict(feature_max_errors=feature_errors,reference_action_max_error=error,finite=bool(np.isfinite(action).all()))
        del runtime;torch.cuda.empty_cache()
    Path('/root/autodl-tmp/F3_control_validation/smoke.json').write_text(json.dumps(dict(status='PASS',variants=results,
        real_data='first Dev12 observation only; Final not read',hardware='NOT_CONNECTED'),indent=2))
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
