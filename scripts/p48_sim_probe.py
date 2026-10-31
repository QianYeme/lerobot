"""P4.8 reset/step/render/schema feasibility; three no-policy smoke rollouts only."""
import os
os.environ['MUJOCO_GL']='egl'
os.environ['PYOPENGL_PLATFORM']='egl'
os.environ['MUJOCO_EGL_DEVICE_ID']='0'

import fcntl
import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path
import sys
import time

ROOT=Path('/root/autodl-tmp/lerobot-main')
OUT=ROOT/'experiments/p4_public_benchmark/p4_8_libero_pilot/index1/result'
DATA=Path('/root/autodl-tmp/p48_data/lerobot_libero')
ASSETS=Path('/root/autodl-tmp/p48_data/libero_assets')


def write(name,value):
    (OUT/name).parent.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n')


def main():
    lock=(OUT/'sim_probe.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert (OUT/'assets_fetch_complete.json').exists() and (OUT/'data_fetch_complete.json').exists()
    import yaml
    package=Path(distribution('hf-libero').locate_file('libero/libero'))
    assert str(package).startswith('/root/autodl-tmp/p48_isolated_env/')
    config=OUT/'libero_config';config.mkdir(exist_ok=True)
    os.environ['LIBERO_CONFIG_PATH']=str(config)
    (config/'config.yaml').write_text(yaml.safe_dump(dict(benchmark_root=str(package),bddl_files=str(package/'bddl_files'),
        init_states=str(package/'init_files'),datasets=str(DATA),assets=str(ASSETS))))
    if not (package/'assets').exists():(package/'assets').symlink_to(ASSETS,target_is_directory=True)
    sys.path.insert(0,str(OUT/'workspace/lerobot/src'))
    import numpy as np
    import pandas as pd
    import torch
    import imageio.v2 as imageio
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv,get_libero_dummy_action
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor.env_processor import LiberoProcessorStep
    torch.set_num_threads(4)
    suite=benchmark.get_benchmark_dict()['libero_object']()
    descriptions=[dict(task_id=i,name=suite.get_task(i).name,instruction=suite.get_task(i).language) for i in range(suite.n_tasks)]
    write('suite_task_mapping.json',descriptions)
    task=suite.get_task(0)
    tasks=pd.read_parquet(DATA/'meta/tasks.parquet')
    normalize=lambda s:' '.join(s.lower().split())
    matching=[int(row.task_index) for name,row in tasks.iterrows() if normalize(str(name))==normalize(task.language)]
    assert len(matching)==1,'Simulator task must exactly match public dataset instruction'
    table=pd.concat([pd.read_parquet(p) for p in sorted((DATA/'data').rglob('*.parquet'))],ignore_index=True)
    selected=table[table.task_index==matching[0]].sort_values(['episode_index','frame_index'])
    ids=sorted(map(int,selected.episode_index.unique()));assert len(ids)>=10
    state=np.stack(selected['observation.state']);action=np.stack(selected.action)
    assert state.shape[1:]==(8,) and action.shape[1:]==(7,)
    assert np.isfinite(state).all() and np.isfinite(action).all() and np.abs(action).max()<=1.00001
    episode_checks=[]
    for ep,frame in selected.groupby('episode_index'):
        assert np.array_equal(frame.frame_index.to_numpy(),np.arange(len(frame)))
        assert np.allclose(np.diff(frame.timestamp.to_numpy()),.1,atol=1e-5)
        episode_checks.append(dict(episode=int(ep),frames=len(frame),first_state=frame.iloc[0]['observation.state'].tolist(),
                                   first_action=frame.iloc[0].action.tolist()))
    write('selected_task.json',dict(suite='libero_object',task_id=0,instruction=task.language,name=task.name,
         dataset_task_index=matching[0],episode_ids=ids,frames=len(selected),selection_rule='First canonical simulator task, exact instruction match; before any model performance',
         state_min=state.min(0).tolist(),state_max=state.max(0).tolist(),action_min=action.min(0).tolist(),action_max=action.max(0).tolist()))
    write('episode_sanity.json',episode_checks)
    processor=LiberoProcessorStep()
    def vectorize(value):
        if isinstance(value,dict):return {k:vectorize(v) for k,v in value.items()}
        return value[None] if isinstance(value,np.ndarray) else value
    def process(obs):
        result=processor._process_observation(preprocess_observation(vectorize(obs)))
        assert result['observation.state'].shape==(1,8)
        for camera in ['image','image2']:
            tensor=result['observation.images.'+camera]
            assert tensor.shape==(1,3,256,256) and tensor.dtype==torch.float32
        return result
    first=None;rows=[]
    smoke=OUT/'sim_smoke';smoke.mkdir(exist_ok=True)
    for initial in [0,0,1,2]:
        env=LiberoEnv(suite,0,'libero_object',obs_type='pixels_agent_pos',episode_length=20,
                      observation_width=256,observation_height=256,init_states=True,episode_index=initial,control_mode='relative')
        try:
            assert env._env.env.hard_reset
            obs,info=env.reset(seed=4701);processed=process(obs)
            snapshot={key:value.detach().cpu().numpy().copy() for key,value in processed.items()}
            if first is None:
                first=snapshot
                write('sim_controller_contract.json',dict(control_freq=env._env.env.control_freq,hard_reset=True,mode=env.control_mode,
                     action_low=env.action_space.low.tolist(),action_high=env.action_space.high.tolist(),
                     controller_input_min=env._env.robots[0].controller.input_min.tolist(),
                     controller_input_max=env._env.robots[0].controller.input_max.tolist(),
                     controller_output_min=env._env.robots[0].controller.output_min.tolist(),
                     controller_output_max=env._env.robots[0].controller.output_max.tolist(),
                     requested_backend='egl',effective_backend=os.environ.get('MUJOCO_GL')))
                continue
            if initial==0:
                difference={key:float(np.abs(value.astype(float)-first[key].astype(float)).max()) for key,value in snapshot.items()}
                write('paired_reset_check.json',dict(status='PASS' if all(v==0 for v in difference.values()) else 'NON_BITWISE',seed=4701,initial_id=0,max_differences=difference))
            frames=[];trajectory=[]
            for step in range(20):
                obs,reward,terminated,truncated,info=env.step(np.array(get_libero_dummy_action(),dtype=np.float32))
                processed=process(obs);render=env.render();assert render.shape==(256,256,3)
                pixels=[(processed['observation.images.'+camera][0].permute(1,2,0).numpy()*255).round().astype(np.uint8) for camera in ['image','image2']]
                frames.append(np.concatenate(pixels,axis=1))
                trajectory.append(dict(step=step,reward=float(reward),terminated=bool(terminated),truncated=bool(truncated),
                     success=bool(info['is_success']),state=processed['observation.state'][0].tolist()))
                if terminated or truncated:break
            imageio.mimsave(smoke/f'init_{initial}_noop.mp4',frames,fps=20,codec='libx264')
            write(f'sim_smoke/init_{initial}_steps.json',trajectory)
            rows.append(dict(init_id=initial,seed=4701,steps=len(trajectory),reported_success=bool(info['is_success']),
                             controller='NO_POLICY_DUMMY_ARM_ZERO_GRIPPER_OPEN',not_a_model_evaluation=True))
        finally:env.close()
    write('sim_smoke_summary.json',dict(status='RESET_STEP_RENDER_PASS',rollouts=rows,policy_training=False,robot=False))
    print('SIM_SMOKE_PASS',task.name,matching[0],len(ids),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        (OUT/'sim_probe_error.log').write_text(traceback.format_exc())
        write('sim_smoke_summary.json',dict(status='ENV_OR_DATA_BLOCKED',policy_training=False))
        raise
