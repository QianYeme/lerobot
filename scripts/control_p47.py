"""Run one supervised P4.7 trial from an independently downloaded model package."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def main():
    early=argparse.ArgumentParser(add_help=False)
    early.add_argument('--bundle',type=Path,required=True)
    boot,_=early.parse_known_args()
    root=boot.bundle.resolve()
    sys.path.insert(0,str(root/'source/src'))
    sys.path.insert(0,str(root))
    from lerobot.scripts.control_f5 import parser,settings,run_trial
    from p47_runtime import Runtime
    identity=json.loads((root/'manifest.json').read_text());role=identity['role']
    parser=parser([role])
    parser.add_argument('--local-on',action='store_true')
    parser.add_argument('--shadow-no-cup',action='store_true')
    parser.add_argument('--light',choices=['bright','dim'],required=True)
    parser.add_argument('--water',choices=['half','empty','absent'],required=True)
    parser.add_argument('--repeat',type=int,choices=[1,2,3],default=1)
    parser.add_argument('--trial-id',dest='trial_id_override')
    parser.add_argument('--robot.use_degrees',dest='robot_use_degrees',choices=['true'],default='true')
    # None means the model native default; ACT guide explicitly supplies1/.01.
    parser.set_defaults(policy_n_action_steps=None,policy_temporal_ensemble_coeff=None)
    args=parser.parse_args();args.robot_use_degrees=True
    if (args.water=='absent') != args.shadow_no_cup:
        parser.error('Absent cup requires --shadow-no-cup; no executed empty-scene trials')
    if role=='VDP-S1' and (args.policy_n_action_steps is not None or args.policy_temporal_ensemble_coeff is not None):
        parser.error('Diffusion uses native8 execution steps and no ACT ensemble flags')
    validated=copy.copy(args);validated.policy_n_action_steps=1;validated.policy_temporal_ensemble_coeff=.01
    common=settings(validated)
    runtime=Runtime(root,args.policy_device,args.policy_n_action_steps,args.policy_temporal_ensemble_coeff,args.local_on)
    common.update(n_action_steps=runtime.policy.config.n_action_steps,
                  temporal_ensemble_coeff=getattr(runtime.policy.config,'temporal_ensemble_coeff',None),
                  light=args.light,water=args.water,repeat=args.repeat,local_on=args.local_on,
                  repo_revision=identity.get('download_revision','SEE_DOWNLOAD_RECEIPT'))
    args.session_dir.mkdir(parents=True,exist_ok=True)
    if args.dry_run:
        import numpy as np
        obs={key:0. for key in runtime.fields+['gripper.load']}
        obs.update(top=np.zeros((480,640,3),dtype=np.uint8),gripper=np.zeros((480,640,3),dtype=np.uint8))
        action,_,_=runtime.step(obs);runtime.reset()
        (args.session_dir/(role+'_dry_run.json')).write_text(json.dumps(dict(action=action.tolist(),common=common,status='SYNTHETIC_SMOKE_NOT_ROBOT'),indent=2))
        return
    if args.shadow_no_cup:
        from lerobot.robots.so_follower import SO101Follower,SO101FollowerConfig
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        robot=SO101Follower(SO101FollowerConfig(port=args.robot_port,id=args.robot_id,use_degrees=True,
            calibration_dir=args.calibration_dir,max_relative_target=args.robot_max_relative_target,
            cameras={name:OpenCVCameraConfig(**{k:v for k,v in c.items() if k!='type'}) for name,c in common['cameras'].items()}))
        path=args.session_dir/(args.trial_id_override or f'{role}_shadow_{time.time_ns()}' )
        path.mkdir()
        try:
            robot.connect(calibrate=False);start=time.perf_counter();runtime.reset()
            with (path/'shadow.jsonl').open('w') as stream:
                while time.perf_counter()-start<common['episode_time_s']:
                    action,_,_=runtime.step(robot.get_observation())
                    stream.write(json.dumps(dict(t=time.time(),predicted_action=action.tolist(),executed=False))+'\n')
        finally:
            runtime.reset();robot.disconnect()
        return
    plan=json.loads((root/'trial_plan.json').read_text())
    selected=next(row for row in plan if row['role']==role and row['local_on']==args.local_on and
                  row['light']==args.light and row['water']==args.water and row['position']==args.position and row['repeat']==args.repeat)
    protocol=SimpleNamespace(SCHEDULE=plan,STAGE_PREFIX='P',
        STAGE_NAMES=['approach','grasp','lift','hold','transport','place','release'],
        trial_id=lambda *_:f'P47_{role}_{args.light}_{args.water}_{args.position}_R{args.repeat}_{"ON" if args.local_on else "OFF"}',
        slot=lambda *_:selected['slot'])
    run_trial(args,runtime,common,protocol=protocol)


if __name__=='__main__':
    main()
