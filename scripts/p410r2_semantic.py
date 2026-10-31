"""Replay logged on-policy actions, verify states/actions, generate synchronized DINO maps."""
import json
import os
from pathlib import Path
import sys

ROOT=Path('/root/autodl-tmp/lerobot-main')
R5=ROOT/'experiments/p4_public_benchmark/p4_8_libero_pilot/index5'
R0=ROOT/'experiments/p4_public_benchmark/p4_10_gate_sweep/index1'
WORK=R0.parent/'index3';OUT=WORK/'result/semantic_review'
sys.path.insert(0,str(R5/'workspace/scripts'))
import p48_r5_common as old
import p48_r5_evaluate as original

def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor.env_processor import LiberoProcessorStep
    os.environ['LIBERO_CONFIG_PATH']=str(original.OLD/'libero_config')
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv
    OUT.mkdir(parents=True,exist_ok=True)
    cfg,policy=old.make_policy('fixed')
    path=R0/'result/train/G5e-2/checkpoints/005000/resume.pt'
    saved=torch.load(path,map_location='cpu',weights_only=False);assert saved['step']==5000
    policy.load_state_dict(saved['model'],strict=True);policy.eval();del saved
    stats=torch.load(R5/'result/train35_statistics.pt',weights_only=True)
    pre,post=make_act_pre_post_processors(cfg,stats);processor=LiberoProcessorStep()
    suite=benchmark.get_benchmark_dict()['libero_object']()
    report=[]
    for init in [35,38,42,43,48]:
        folder=OUT/f'init_{init}';folder.mkdir(exist_ok=True)
        source=R0/f'result/rollouts/G5e-2/rollouts/fixed/init_{init:02d}_trajectory.json'
        trajectory=json.loads(source.read_text());selected={round(q*(len(trajectory)-1)) for q in [0,.15,.3,.5,.7,.9,1.]}
        env=LiberoEnv(suite,0,'libero_object',obs_type='pixels_agent_pos',init_states=True,episode_index=init,
            control_mode='relative',observation_width=256,observation_height=256,episode_length=280,num_steps_wait=10)
        max_state=max_action=0.;rows=[]
        try:
            obs,_=env.reset(seed=1);policy.reset();pre.reset();post.reset()
            for step,record in enumerate(trajectory):
                raw=processor._process_observation(preprocess_observation(original.vectorize(obs)))
                state_error=float(np.max(np.abs(raw['observation.state'][0].numpy()-record['state_before'])))
                max_state=max(max_state,state_error);assert state_error<=1e-6,(init,step,state_error)
                top=raw['observation.images.image'].clone().cuda();batch=pre(raw);batch['_oc_raw_top']=top
                with torch.inference_mode():
                    action=post(policy.select_action(batch)).cpu().numpy()[0]
                error=float(np.max(np.abs(action-record['action'])));max_action=max(max_action,error)
                assert error<=1e-6,(init,step,error)
                if step in selected:
                    with torch.inference_mode():
                        oc=policy.model.object_conditioning;patches,grid=oc.dino(top)
                        sim=(F.normalize(patches,dim=-1)@oc.prototype_bank.prototypes.T).max(-1).values.reshape(*grid)
                        weights=torch.softmax(sim.flatten()/.2,dim=0).reshape(*grid)
                    image=top[0].permute(1,2,0).cpu().numpy();heat=weights.cpu().numpy()
                    np.savez(folder/f'frame_{step:04d}_maps.npz',similarity=sim.cpu().numpy(),weights=heat,rgb=image)
                    fig,axes=plt.subplots(1,3,figsize=(10,3))
                    axes[0].imshow(image);axes[1].imshow(sim.cpu(),cmap='viridis')
                    axes[2].imshow(image);axes[2].imshow(heat,extent=[0,256,256,0],alpha=.6,cmap='magma')
                    for ax in axes:ax.axis('off')
                    axes[0].set_title(f'init{init} step{step}');axes[1].set_title('DINO patch similarity');axes[2].set_title('global softmax overlay')
                    fig.tight_layout();fig.savefig(folder/f'frame_{step:04d}.png');plt.close(fig)
                    rows.append(dict(step=step,phase='TIME_FRACTION_ONLY_NOT_GT_STAGE',patch_grid=list(grid),image_size=[256,256],
                        coordinate_map='patch row/col -> pixel center ((col+.5)*16,(row+.5)*16)',
                        global_xy=oc.tokenizer(patches,grid,oc.prototype_bank.prototypes)[0][0,0,-3:].cpu().tolist()))
                obs,_,terminated,truncated,_=env.step(np.asarray(record['action']))
        finally:env.close()
        entry=dict(init=init,max_state_error=max_state,max_action_error=max_action,frames=len(trajectory),
            checkpoint_sha=old.sha(path),prototype_sha=old.sha(R5/'result/prototype_bank.pt'),trajectory_sha=old.sha(source),
            replay='EXACT_WITHIN_1e-6',maps=rows,no_gt_segmentation=True)
        old.write(folder/'replay_audit.json',entry);report.append(entry)
    old.write(OUT/'replay_complete.json',dict(status='DONE',cases=report,review='EXECUTOR_VISUAL_REVIEW_PENDING'))

if __name__=='__main__':main()
