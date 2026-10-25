"""Server-only export and real-RGB parity for the one authorized historical133k."""
import gc
import json
import os
import shutil
from pathlib import Path
import numpy as np
import torch
from lerobot.utils.r5_11_protocol import verify_bundle,R3_SHA,FULL60_NORM_SHA,POSITION_FIELDS,digest
from lerobot.policies.act.r5_11_runtime import R511Runtime

W=Path('/root/autodl-tmp/R5_11_20261025')
OLD=Path('/root/autodl-tmp/R5_3_20261024')


def main():
    source=OLD/'runs/G60-S3-133k/attempt1/checkpoints/133000/pretrained_model'
    record=next(x for x in json.loads((OLD/'evaluation/checkpoint_identity.json').read_text()) if x['model']=='G60-S3-133k')
    complete=json.loads((source.parents[2]/'complete.json').read_text())
    assert complete['sha256']==record['sha256']==R3_SHA==digest(source/'model.safetensors')
    assert complete['step']==133000 and record['formal'] is True
    assert digest(source/'f2_1_normalization.json')==FULL60_NORM_SHA
    bundle=W/'bundle'
    if not bundle.exists():
        bundle.mkdir(exist_ok=False)
        target=bundle/'models/R3';target.mkdir(parents=True)
        for name in ['model.safetensors','config.json','f2_1_normalization.json']:os.link(source/name,target/name)
        (bundle/'normalization').mkdir();shutil.copy2(source/'f2_1_normalization.json',bundle/'normalization/full60.json')
        old=Path('/root/autodl-tmp/R5_10_20261025/bundle')
        for name in ['assets','torch_cache']:shutil.copytree(old/name,bundle/name,copy_function=os.link)
        identity=dict(global_path='models/R3',global_sha256=R3_SHA,source_step=133000,seed=3,source_checkpoint=str(source),
            normalization_path='normalization/full60.json',state_fields=POSITION_FIELDS,global_load=True,local_path=None,
            source_experiment='R5.3 historical exposure-matched trajectory; not the R5.10 M1 trajectory')
        (bundle/'model_identity.json').write_text(json.dumps({'R3':identity},indent=2)+'\n')
        (bundle/'manifest.json').write_text(json.dumps(dict(sha256={p.relative_to(bundle).as_posix():digest(p) for p in bundle.rglob('*') if p.is_file()}),indent=2)+'\n')
    verify_bundle(bundle)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.hub.set_dir(str(bundle/'torch_cache/hub'))
    runtime=R511Runtime(bundle,'R3','cuda',1,.01);maximum=0.;native_max=0.;samples=0
    raw=Path('/root/autodl-tmp/R5_1_20261023/log_based_sequence_analysis/common_input_raw')
    for path in sorted(raw.glob('input_ep*.npz')):
        ep=int(path.stem.split('ep')[-1])
        with np.load(path) as z:data={k:z[k].copy() for k in z.files}
        state=data['state'];obs=dict(zip(POSITION_FIELDS,map(float,state[:6])))
        obs.update({'gripper.load':float(state[6]),'gripper.curr':float(state[7]),'top':data['top'],'gripper':data['gripper']})
        batch=runtime.prepare(obs)
        for key,value in [('observation.state','state60'),('observation.load','load60'),('observation.images.top','top_model_tensor'),('observation.images.gripper','gripper_model_tensor')]:assert torch.equal(batch[key].cpu(),torch.from_numpy(data[value])),key
        with np.load(OLD/'evaluation/raw'/f'G60-S3-133k_ep{ep:03d}_native.npz') as z:expected=z['normalized_chunk'].copy();native=z['native_chunk'].copy()
        with torch.inference_mode():actual=runtime.policy.predict_action_chunk(batch)[0].cpu().numpy()
        maximum=max(maximum,float(np.abs(actual-expected).max()))
        runtime.reset();action,nominal,telemetry=runtime.step(obs)
        native_max=max(native_max,float(np.abs(nominal-native[0]).max()))
        assert telemetry is None and runtime.local is None and np.array_equal(action,nominal)
        runtime.reset();assert np.array_equal(action,runtime.step(obs)[0])
        samples+=1
    assert samples==36 and maximum<=1e-6 and native_max<=1e-5,(samples,maximum,native_max)
    result=dict(status='PASS_OFFLINE_ONLY',samples=samples,normalized_chunk_max_difference=maximum,first_step_native_max_difference=native_max,input_bitwise_equal=True,reset_repeat_difference=0,
        original_checkpoint_identity=record,robot_trials=0,R5_11_COMPLETE=False,scope='Training-input parity only; no held-out or real-robot claim')
    (W/'runtime_smoke.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
