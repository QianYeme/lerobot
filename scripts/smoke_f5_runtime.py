"""Server GPU real-Dev smoke for all F5 paths; no robot execution/import."""
import gc
import ast
import json
from pathlib import Path

import numpy as np
import torch

from run_f4_1_worker import Global
from lerobot.policies.act.f5_runtime import F5Runtime
from lerobot.utils.f5_protocol import MODELS,POSITION_FIELDS,BOUND


def main():
    work=Path('/root/autodl-tmp/F5_20261022');bundle=work/'bundle'
    torch.hub.set_dir(str(bundle/'torch_cache/hub'));torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=True
    exp=Global(1);loader=exp.loader('dev',workers=0);raw=next(iter(loader))
    state=raw['observation.state'][0].cpu().numpy();assert state.shape==(8,),state.shape
    obs=dict(zip(POSITION_FIELDS,map(float,state[:6])));obs['gripper.load']=float(state[6]);obs['gripper.curr']=float(state[7])
    for name in ['top','gripper']:
        obs[name]=(raw['observation.images.'+name][0].permute(1,2,0).numpy()*255).round().astype(np.uint8)
    report={}
    for model in MODELS:
        runtime=F5Runtime(bundle,model,'cuda',1,.01)
        batch=runtime.prepare(obs)
        with torch.inference_mode():
            expected=runtime.policy.predict_action_chunk(batch)[:,0]
            expected=runtime.post(expected).to('cuda') if runtime.post else expected*runtime.std['action']+runtime.mean['action']
            runtime.reset();action,nominal,telemetry=runtime.step(obs)
        np.testing.assert_allclose(nominal,expected[0].cpu().numpy(),rtol=1e-5,atol=1e-4)
        assert np.array_equal(action[:5],nominal[:5]) and np.isfinite(action).all()
        if runtime.local:
            delta=float(telemetry['clipped_delta'][0]);assert abs(delta)<=BOUND+1e-6
            assert abs(float(action[5]-nominal[5])-delta)<1e-5
            with torch.inference_mode():
                local=runtime.local
                expected_delta=local(torch.tensor([obs['gripper.load']],device='cuda'),torch.tensor([obs['gripper.pos']],device='cuda'),torch.tensor([nominal[5]],device='cuda'))
            torch.testing.assert_close(telemetry['clipped_delta'],expected_delta.clamp(-BOUND,BOUND))
        else:np.testing.assert_array_equal(action,nominal)
        runtime.step(obs);runtime.reset();again,_,_=runtime.step(obs)
        np.testing.assert_allclose(action,again,rtol=1e-5,atol=1e-4)
        if model!='P1-Hist':
            changed=dict(obs);changed['gripper.curr']=1e9;runtime.reset();alternate,_,_=runtime.step(changed)
            np.testing.assert_array_equal(again,alternate)
        report[model]=dict(status='PASS_REAL_DEV_FIRST_FRAME',state_width=int(batch['observation.state'].shape[-1]),
            global_load='observation.load' in batch,local=runtime.local is not None,
            first_step_nominal_matches_raw_h0=True,residual_applied_once=True,reset_reproduces_first_step=True,
            nonhistorical_current_invariance='PASS' if model!='P1-Hist' else 'HISTORICAL_CURRENT_INPUT_RETAINED')
        del runtime,batch;gc.collect();torch.cuda.empty_cache()
    result=dict(status='PASS_OFFLINE_ONLY',models=report,device='RTX5090 server GPU',hardware='NOT_CONNECTED',
        final12_reads=0,robot_trials=0,note='Temporal ensemble first/reset step verified; not a robot safety or closed-loop validation')
    import inspect
    from lerobot.policies.act.modeling_act import ACTPolicy
    active=Path(inspect.getfile(ACTPolicy));parent=Path('/root/autodl-tmp/F2_1_r1_20261019/source/src/lerobot/policies/act/modeling_act.py')
    result['current_vs_training_ACT_ast_equal']=ast.dump(ast.parse(active.read_text(encoding='utf-8')))==ast.dump(ast.parse(parent.read_text(encoding='utf-8')))
    result['ACT_source']=str(active)
    (work/'runtime_smoke.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
