"""One supervised P4.3 trial; no automatic robot queue or persistent STOP locks."""
import csv
from types import SimpleNamespace

from lerobot.scripts.control_f5 import parser, settings, run_trial, save


def main():
    p = parser(['L1', 'F3', 'A1', 'L1_160K'])
    next(a for a in p._actions if a.dest == 'position').choices = [f'P{i:02d}' for i in range(1,7)]
    p.add_argument('--repeat', type=int, choices=[1,2], default=1)
    p.add_argument('--trial-id', dest='trial_id_override')
    p.add_argument('--robot.use_degrees', dest='robot_use_degrees', choices=['true','false'], default='true')
    args = p.parse_args()
    if args.robot_use_degrees != 'true':
        p.error('Frozen native interface requires use_degrees=true; do not silently convert units')
    args.robot_use_degrees = True
    common = settings(args)
    common.update(repeat=args.repeat, use_degrees=True, start_method='manual_reset',
                  preprocessing={'L1':'LEGACY_DOUBLE_NORM','L1_160K':'LEGACY_DOUBLE_NORM',
                                 'F3':'FIXED_SINGLE_NORM','A1':'NO_DINO'}[args.model])
    import torch
    import numpy as np
    from lerobot.policies.act.p43_runtime import P43Runtime
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    runtime=P43Runtime(args.bundle,args.model,args.policy_device,args.policy_n_action_steps,args.policy_temporal_ensemble_coeff)
    args.session_dir.mkdir(parents=True,exist_ok=True)
    if args.dry_run:
        observation={key:0. for key in runtime.identity['state_fields']}
        observation.update(top=np.zeros((480,640,3),dtype=np.uint8),gripper=np.zeros((480,640,3),dtype=np.uint8))
        action,_,_=runtime.step(observation)
        runtime.reset()
        save(args.session_dir/f'offline_{args.model}.json',dict(status='SYNTHETIC_RUNTIME_SMOKE_NOT_ROBOT',
             model=args.model,action=action.tolist(),identity=runtime.identity,common=common))
        return
    with (args.bundle/'trial_plan_frozen.csv').open(newline='') as stream:
        plan=list(csv.DictReader(stream))
    selected=next(row for row in plan if row['model']==args.model and row['position']==args.position and int(row['repeat'])==args.repeat)
    protocol=SimpleNamespace(SCHEDULE=plan, STAGE_PREFIX='P',
        STAGE_NAMES=['approach','grasp','lift','hold','transport','place','release'],
        trial_id=lambda model,position:f"P43_{position}_R{args.repeat}_{model}",
        slot=lambda model,position:int(selected['slot']))
    run_trial(args,runtime,common,protocol=protocol)


if __name__=='__main__':
    main()
