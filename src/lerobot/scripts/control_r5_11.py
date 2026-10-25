"""One R5.11 manual-reset trial; reuse existing reference bundles, no auto queue."""
import numpy as np
from lerobot.scripts.control_f5 import parser,settings,run_trial,save
from lerobot.utils import r5_11_protocol as protocol


def main():
    p=parser(protocol.MODELS)
    p.add_argument('--trial-id',dest='trial_id_override')
    p.add_argument('--trial-kind',choices=['stage_a','confirmation','stage_b'],default='stage_a')
    a=p.parse_args();common=settings(a)
    if a.model=='R0' and a.position!='A':p.error('P1 is only a Stage A reference in this entry')
    import torch
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    if (a.bundle/'torch_cache').exists():torch.hub.set_dir(str(a.bundle/'torch_cache/hub'))
    if a.model in ['R0','R1']:
        from lerobot.policies.act.f5_runtime import F5Runtime
        runtime=F5Runtime(a.bundle,'P1-Hist' if a.model=='R0' else 'New-GL',a.policy_device,a.policy_n_action_steps,a.policy_temporal_ensemble_coeff)
    elif a.model=='R2':
        from lerobot.policies.act.r5_10_runtime import R510Runtime
        runtime=R510Runtime(a.bundle,'M1',a.policy_device,a.policy_n_action_steps,a.policy_temporal_ensemble_coeff)
    else:
        from lerobot.policies.act.r5_11_runtime import R511Runtime
        runtime=R511Runtime(a.bundle,'R3',a.policy_device,a.policy_n_action_steps,a.policy_temporal_ensemble_coeff)
    common.update(model_label=protocol.LABELS[a.model],trial_kind=a.trial_kind,start_method='manual_reset_no_numeric_home_gate',
        cudnn_benchmark=False,cudnn_deterministic=True,TF32=True,stages=protocol.STAGE_NAMES)
    a.session_dir.mkdir(parents=True,exist_ok=True)
    if a.dry_run:
        obs={k:0. for k in protocol.POSITION_FIELDS+['gripper.load','gripper.curr']}
        obs.update(top=np.zeros((480,640,3),np.uint8),gripper=np.zeros((480,640,3),np.uint8))
        action,_,_=runtime.step(obs);runtime.reset()
        save(a.session_dir/f'offline_{a.model}.json',dict(status='SYNTHETIC_CODE_SMOKE_ONLY',action=action.tolist(),common=common,identity=runtime.identity,hardware='NOT_CONNECTED'))
        return
    run_trial(a,runtime,common,protocol=protocol)


if __name__=='__main__':main()
