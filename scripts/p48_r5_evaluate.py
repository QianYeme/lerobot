"""Full Dev9 diagnostics and10 native paired LIBERO rollouts per frozen r5 model."""
import argparse
import csv
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID','0')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_DATASETS_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
from p48_r5_common import OLD,OUT,CachedData,make_policy,sha,write


def load(kind):
    import torch
    cfg,policy=make_policy(kind)
    latest=json.loads((OUT/'runs'/kind/'latest.json').read_text())
    assert latest['step']==10000 and sha(latest['path'])==latest['sha256']
    state=torch.load(latest['path'],map_location='cpu',weights_only=False)
    assert state['step']==10000 and state['kind']==kind
    assert state['split_sha256']==sha(OUT/'dataset_split_manifest.json')
    assert state['stats_sha256']==sha(OUT/'train35_statistics.pt')
    if kind!='act':assert state['bank_sha256']==sha(OUT/'prototype_bank.pt')
    policy.load_state_dict(state['model'],strict=True);policy.eval();return cfg,policy,latest


def dev(kind):
    import numpy as np
    import torch
    cfg,policy,latest=load(kind);data=CachedData('dev')
    folder=OUT/'evaluation'/kind;folder.mkdir(parents=True,exist_ok=True)
    predictions=[];targets=[]
    started=time.time()
    with torch.inference_mode():
        for offset in range(0,len(data),16):
            batch=data.batch(range(offset,min(len(data),offset+16)))
            pred=policy.predict_action_chunk(batch)[:,0]
            predictions.append(pred.cpu().numpy());targets.append(batch['action'][:,0].cpu().numpy())
    pred=np.concatenate(predictions);target=np.concatenate(targets)
    stats=data.stats['action'];std=stats['std'].numpy()+1e-8;mean=stats['mean'].numpy()
    normalized=np.abs(pred-target);raw=normalized*std
    episode_ids=data.arrays['episode'];rows=[]
    for ep in sorted(set(episode_ids.tolist())):
        mask=episode_ids==ep
        row=dict(kind=kind,episode=int(ep),samples=int(mask.sum()),h0_arm_normalized_mae=float(normalized[mask,:6].mean()),
            h0_gripper_normalized_mae=float(normalized[mask,6].mean()),h0_arm_raw_mae=float(raw[mask,:6].mean()),
            h0_gripper_raw_mae=float(raw[mask,6].mean()))
        row.update({f'h0_dim{i}_raw_mae':float(raw[mask,i].mean()) for i in range(7)})
        rows.append(row)
    with (folder/'dev9_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez(folder/'dev9_h0_predictions.npz',episode=episode_ids,frame=data.arrays['frame'],
             prediction_normalized=pred,target_normalized=target,prediction_raw=pred*std+mean,target_raw=target*std+mean)
    diagnostics=[]
    if kind!='act':
        oc=policy.model.object_conditioning;trace={}
        def capture_tokens(_module,_args,output):
            trace['token_mean_abs']=float(output[0].abs().mean());trace['similarity_min']=float(output[1].min());trace['similarity_max']=float(output[1].max())
        def capture_residual(_module,args,output):
            trace['residual_to_query_norm_ratio']=float((output-args[0]).norm()/args[0].norm().clamp_min(1e-12))
        handles=[oc.tokenizer.register_forward_hook(capture_tokens),oc.register_forward_hook(capture_residual)]
        with torch.inference_mode():
            for ep in sorted(set(episode_ids.tolist())):
                indices=np.flatnonzero(episode_ids==ep)
                for index in [indices[0],indices[len(indices)//2],indices[-1]]:
                    batch=data.batch([index]);on=policy.predict_action_chunk(batch)[:,0];values=dict(trace)
                    gate=oc.gate.detach().clone()
                    try:
                        oc.gate.zero_();off=policy.predict_action_chunk(batch)[:,0]
                    finally:oc.gate.copy_(gate)
                    delta=(on-off).abs()[0].cpu().numpy()
                    diagnostics.append(dict(kind=kind,episode=int(ep),frame=int(data.arrays['frame'][index]),gate=float(gate),
                        h0_on_off_arm_normalized_mean_abs=float(delta[:6].mean()),h0_on_off_gripper_normalized_abs=float(delta[6]),
                        h0_on_off_arm_raw_mean_abs=float((delta*std)[:6].mean()),h0_on_off_gripper_raw_abs=float((delta*std)[6]),**values))
        for handle in handles:handle.remove()
        with (folder/'object_path_diagnostics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(diagnostics[0]));w.writeheader();w.writerows(diagnostics)
    metrics={key:float(np.mean([r[key] for r in rows])) for key in rows[0] if key not in ['kind','episode','samples']}
    write(folder/'dev_complete.json',dict(status='FULL_DEV9_H0_COMPLETE',kind=kind,episodes=len(rows),samples=len(data),
        checkpoint=latest,episode_equal_metrics=metrics,elapsed_s=time.time()-started,object_on_off_samples=len(diagnostics),
        metric_units='normalized Train35 z-score / raw OSC controller input, not joint degrees',
        teacher_forced=True,not_a_closed_loop_success_rate=True,offline_local_files=True))
    print('FULL_DEV9_DONE',kind,len(data),flush=True)


def vectorize(obs):
    import numpy as np
    return {key:vectorize(value) if isinstance(value,dict) else value[None] if isinstance(value,np.ndarray) else value for key,value in obs.items()}


def rollout(kind):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor.env_processor import LiberoProcessorStep
    os.environ['LIBERO_CONFIG_PATH']=str(OLD/'libero_config')
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv
    cfg,policy,latest=load(kind)
    stats=torch.load(OUT/'train35_statistics.pt',weights_only=True)
    pre,post=make_act_pre_post_processors(cfg,stats);processor=LiberoProcessorStep()
    contract=json.loads((OUT/'timing_contract.json').read_text());suite=benchmark.get_benchmark_dict()['libero_object']()
    folder=OUT/'rollouts'/kind;folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for initial,seed in zip(contract['paired_initial_ids'],contract['episode_seeds']):
        env=LiberoEnv(suite,0,'libero_object',obs_type='pixels_agent_pos',init_states=True,
            episode_index=initial,control_mode='relative',observation_width=256,observation_height=256,
            episode_length=contract['sim_max_steps'],num_steps_wait=contract['reset_wait_steps'])
        video=folder/f'init_{initial:02d}.mp4';trajectory=[];latencies=[];started=time.time();success=False
        writer=imageio.get_writer(str(video),fps=20,codec='libx264',quality=7,macro_block_size=16)
        try:
            observation,info=env.reset(seed=seed);assert env._env.env.control_freq==20
            policy.reset();pre.reset();post.reset()
            for step in range(contract['sim_max_steps']):
                raw=processor._process_observation(preprocess_observation(vectorize(observation)))
                assert raw['observation.state'].shape==(1,8)
                cameras=[(raw['observation.images.'+key][0].permute(1,2,0).numpy()*255).round().astype(np.uint8) for key in ['image','image2']]
                writer.append_data(np.concatenate(cameras,axis=1))
                top=raw['observation.images.image'].clone().cuda();batch=pre(raw);batch['_oc_raw_top']=top
                with torch.inference_mode():
                    torch.cuda.synchronize();start=time.perf_counter();normalized=policy.select_action(batch)
                    action=post(normalized);torch.cuda.synchronize();latencies.append(time.perf_counter()-start)
                action=action.detach().cpu().numpy()[0];assert action.shape==(7,) and np.isfinite(action).all()
                before=raw['observation.state'][0].tolist()
                observation,reward,terminated,truncated,info=env.step(action)
                success=bool(info['is_success'])
                trajectory.append(dict(step=step,state_before=before,action=action.tolist(),reward=float(reward),
                    success=success,terminated=bool(terminated),truncated=bool(truncated),
                    native_auto_reset_after_terminal=bool(terminated)))
                if terminated or truncated:break
        finally:
            writer.close();env.close()
        write(folder/f'init_{initial:02d}_trajectory.json',trajectory)
        rows.append(dict(kind=kind,init_id=initial,seed=seed,success=success,steps=len(trajectory),
            wall_seconds=time.time()-started,mean_policy_call_seconds=float(np.mean(latencies)),
            max_policy_call_seconds=float(np.max(latencies)),video=str(video),video_sha256=sha(video),
            failure='NONE' if success else 'NO_SUCCESS_WITHIN_NATIVE_280_STEP_BUDGET',
            first_failure_stage='NOT_APPLICABLE' if success else 'UNKNOWN_NO_PHASE_ORACLE',pilot_only=True))
        with (folder/'libero_rollouts.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        print('PAIRED_ROLLOUT',kind,initial,'SUCCESS',success,'STEPS',len(trajectory),flush=True)
    assert len(rows)==10
    write(folder/'rollout_complete.json',dict(status='TEN_PAIRED_ROLLOUTS_COMPLETE',kind=kind,checkpoint=latest,
         episodes=10,successes=sum(r['success'] for r in rows),success_rate=sum(r['success'] for r in rows)/10,
         control_freq=20,action_repeat=1,pilot_only=True,n_action_steps=cfg.n_action_steps,
         temporal_ensemble_coeff=cfg.temporal_ensemble_coeff,init_ids=contract['paired_initial_ids'],episode_seeds=contract['episode_seeds'],offline_local_files=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['act','legacy','fixed'],required=True)
    parser.add_argument('--mode',choices=['dev','rollout'],required=True);args=parser.parse_args()
    if args.mode=='dev':dev(args.kind)
    else:rollout(args.kind)
