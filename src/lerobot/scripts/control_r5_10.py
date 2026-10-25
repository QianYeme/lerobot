"""Direct R5.10 trial; no Stage0/shadow/STOP locks, initial state is recorded."""
import numpy as np

from lerobot.scripts.control_f5 import parser,settings,run_trial,save
from lerobot.utils import r5_10_protocol as protocol


def main():
    p=parser(protocol.MODELS)
    p.add_argument('--trial-id',dest='trial_id_override')
    p.add_argument('--trial-kind',choices=['primary','confirmation','optional'],default='primary')
    a=p.parse_args();common=settings(a)
    import torch
    from lerobot.policies.act.r5_10_runtime import R510Runtime
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    common.update(cudnn_benchmark=False,cudnn_deterministic=True,matmul_tf32=True,cudnn_tf32=True,
        model_label=protocol.LABELS[a.model],trial_kind='optional' if a.model=='M4' else a.trial_kind,
        start_method='manual_reset_no_home_tolerance_gate')
    if (a.bundle/'torch_cache').exists():torch.hub.set_dir(str(a.bundle/'torch_cache/hub'))
    a.session_dir.mkdir(parents=True,exist_ok=True)
    runtime=R510Runtime(a.bundle,a.model,a.policy_device,a.policy_n_action_steps,a.policy_temporal_ensemble_coeff)
    if a.dry_run:
        observation={k:0. for k in protocol.POSITION_FIELDS+['gripper.load','gripper.curr']}
        observation.update(top=np.zeros((480,640,3),np.uint8),gripper=np.zeros((480,640,3),np.uint8))
        action,_,_=runtime.step(observation);runtime.reset()
        save(a.session_dir/f'offline_{a.model}.json',dict(status='SYNTHETIC_CODE_SMOKE_ONLY',action=action.tolist(),common=common,hardware='NOT_CONNECTED'))
        return
    run_trial(a,runtime,common,protocol=protocol)


if __name__=='__main__':main()
