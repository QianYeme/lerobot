"""Real-data GPU check of the R5 deployment path, no hardware connection."""
import gc
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.act.r5_runtime import R5Runtime
from lerobot.utils.r5_protocol import MODELS,POSITION_FIELDS,BOUND
from run_f5_1_r3 import Full60,OLD,checkpoint,load_policy,local_predictions

WORK=Path('/root/autodl-tmp/R5_20261023')


def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.hub.set_dir(str(WORK/'bundle/torch_cache/hub'))
    first=Full60(3);raw=next(iter(first.loader('replay',workers=0)))
    state=raw['observation.state'][0].numpy()
    observation=dict(zip(POSITION_FIELDS,map(float,state[:6])))
    observation.update({'gripper.load':float(state[6]),'gripper.curr':float(state[7])})
    for name in ['top','gripper']:
        observation[name]=(raw['observation.images.'+name][0].permute(1,2,0).numpy()*255).round().astype(np.uint8)
    report={}
    for model in MODELS:
        exp=Full60(4 if model=='M5' else 3,old=model in ['M1','M3'])
        batch=exp.batch({k:v[:1] if torch.is_tensor(v) else v for k,v in raw.items()},'A1')
        batch.pop('action');reference=load_policy(exp,OLD if model in ['M1','M3'] else checkpoint(4 if model=='M5' else 3))
        with torch.inference_mode():expected=reference.predict_action_chunk(batch)[:,0]*exp.std['action']+exp.mean['action']
        runtime=R5Runtime(WORK/'bundle',model,'cuda',1,.01)
        prepared=runtime.prepare(observation)
        for key in ['observation.state','observation.load','observation.images.top','observation.images.gripper']:
            if key=='observation.load':
                mean=runtime.norm['load']['mean'];std=runtime.norm['load']['std'];value=observation['gripper.load']
                online=np.float32((value-mean)/std)
                cached=np.float32((np.float32(value)-np.float32(mean))/np.float32(std))
                assert float(prepared[key][0,0])==float(cached)
                assert float(batch[key][0,0])==float(cached)
                load_difference=abs(float(online)-float(cached))
                assert load_difference<=2*np.finfo(np.float32).eps*max(1,abs(float(online)),abs(float(cached)))
            else:torch.testing.assert_close(prepared[key],batch[key],rtol=0,atol=0)
        with torch.inference_mode():
            reference_online=reference.predict_action_chunk(prepared)[:,0]*exp.std['action']+exp.mean['action']
            runtime_offline=runtime.policy.predict_action_chunk(batch)[:,0]*runtime.std['action']+runtime.mean['action']
            runtime_online=runtime.policy.predict_action_chunk(prepared)[:,0]*runtime.std['action']+runtime.mean['action']
        probe=dict(model=model,input_load_difference=load_difference,
            same_reference_input_effect=float((reference_online-expected).abs().max()),
            same_inputs_model_effect=float((runtime_offline-expected).abs().max()),
            same_runtime_input_effect=float((runtime_online-runtime_offline).abs().max()),
            differing_parameters=[k for k,v in reference.state_dict().items() if not torch.equal(v,runtime.policy.state_dict()[k])])
        (WORK/f'parity_probe_{model}.json').write_text(json.dumps(probe,indent=2)+'\n')
        print('PARITY_PROBE',json.dumps(probe),flush=True)
        action,nominal,telemetry=runtime.step(observation)
        np.testing.assert_allclose(nominal,expected[0].cpu().numpy(),rtol=0,atol=.001)
        np.testing.assert_array_equal(action[:5],nominal[:5])
        if runtime.local:
            data=dict(state=np.asarray([state[:6]]),raw_load=np.asarray([state[6]]),prediction_native=np.asarray([nominal]),target_native=np.zeros((1,6)))
            composed,delta=local_predictions(3,data,old=model=='M3')
            np.testing.assert_allclose(action,composed[0],rtol=0,atol=.001)
            assert abs(float(telemetry['clipped_delta'][0]))<=BOUND+1e-6
        repeated=[]
        for _ in range(3):runtime.reset();repeated.append(runtime.step(observation)[0])
        for value in repeated:np.testing.assert_array_equal(action,value)
        changed=dict(observation);changed['gripper.curr']=1e9
        runtime.reset();np.testing.assert_array_equal(action,runtime.step(changed)[0])
        report[model]=dict(status='PASS_REAL_RGB_BATCH1',normalization_path=runtime.identity['normalization_path'],
            image_state_bitwise_matches_reference=True,actual_load_input_max_difference=0.0,legacy_vs_R4_load_difference=load_difference,
            load_precision_reason='R5 uses exact R4 float32 arithmetic; difference here records rejected legacy float64-rounded route',
            nominal_max_native_difference=float(np.abs(nominal-expected[0].cpu().numpy()).max()),
            arm_preserved=True,local_reference_checked=runtime.local is not None,reset_difference=0,current_ignored=True)
        del runtime,reference,batch,prepared;gc.collect();torch.cuda.empty_cache()
    result=dict(status='PASS_OFFLINE_ONLY',models=report,cudnn_benchmark=False,matmul_tf32=True,cudnn_tf32=True,
        n_action_steps=1,temporal_ensemble_coeff=.01,robot_trials=0,
        scope='Real RGB batch1 first/reset step and local composition; not closed-loop robot evidence')
    (WORK/'runtime_smoke.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
