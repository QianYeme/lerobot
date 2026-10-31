"""Same frozen Dev15 samples for Legacy P1 seed1 80/120/160k; no robot or Final15."""
import argparse
import json
import time
import sys
from pathlib import Path

FROZEN_SRC=Path('/root/autodl-tmp/lerobot-main/experiments/p4_final_p1_training/p4_5_robot_reliability_server10h/index1/newgl_code/src')
if FROZEN_SRC.exists():sys.path.insert(0,str(FROZEN_SRC))

import numpy as np
import pandas as pd
import torch

import p4_1_pilot as old
import p45_vanilla_worker as common
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run');parser.add_argument('--seed');parser.add_argument('--deadline',type=float,required=True)
    a=parser.parse_args();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    out=common.OUT
    ids=json.loads((common.PRIOR/'dev15_episode_ids.json').read_text());dataset=old.Mapped(ids)
    meta=pd.read_parquet(common.DATA/'meta/episodes/chunk-000/file-000.parquet',filters=[('episode_index','in',ids)])
    p41=common.ROOT/'experiments/p4_final_p1_training/p4_1_unattended_p1_pilot/index3/result'
    p42=common.ROOT/'experiments/p4_final_p1_training/p4_2_unattended_multi_model/index1/result/runs/P0_P1_S1_160k'
    stats=torch.load(p41/'train90_statistics.pt',weights_only=False)
    for step in [80000,120000,160000]:
        if time.time()>=a.deadline:break
        source=p41/'checkpoints/pilot/080000/resume.pt' if step==80000 else p42/'checkpoints/formal'/f'{step:06d}'/'resume.pt'
        if not source.exists():
            common.append(out/'legacy_missing_checkpoints.csv',dict(step=step,path=str(source),status='MISSING_NOT_EVALUATED'));continue
        cfg=PreTrainedConfig.from_pretrained(source.parent/'pretrained_model')
        cfg.object_conditioning_prototype_path=str(p41/'formal6_train90_prototypes.pt')
        policy=ACTPolicy(cfg).cuda().eval().requires_grad_(False)
        saved=torch.load(source,map_location='cpu',weights_only=False);assert saved['step']==step;policy.load_state_dict(saved['model']);del saved
        pre,post=make_act_pre_post_processors(cfg,stats);rows=[];offset=0
        for ep in ids:
            length=int(meta[meta.episode_index==ep].iloc[0].length)
            for fraction in [0,.15,.3,.5,.7,.9]:
                batch=pre(torch.utils.data.default_collate([dataset[offset+min(length-1,int(fraction*length))]]))
                with torch.inference_mode():
                    torch.cuda.synchronize();began=time.perf_counter();pred=policy.predict_action_chunk(batch);torch.cuda.synchronize();latency=time.perf_counter()-began
                    gate=policy.model.object_conditioning.gate;alpha=gate.detach().clone();gate.zero_()
                    try:off=policy.predict_action_chunk(batch)
                    finally:gate.copy_(alpha)
                    valid=~batch['action_is_pad'][0];error=(pred[0]-batch['action'][0]).abs()[valid];physical=error*stats['action']['std'].cuda()
                row=dict(step=step,episode=ep,progress_proxy=fraction,arm_normalized=float(error[:,:5].mean()),
                    gripper_normalized=float(error[:,5].mean()),arm_raw=float(physical[:,:5].mean()),gripper_raw=float(physical[:,5].mean()),
                    h0_gripper_normalized=float((pred[0,0,5]-batch['action'][0,0,5]).abs()),gate=float(alpha),
                    object_on_off=float((pred-off).abs().mean()),inference_ms=1000*latency)
                common.append(out/'legacy_checkpoint_per_sample.csv',row);rows.append(row)
            offset+=length
        frame=pd.DataFrame(rows)
        common.append(out/'legacy_checkpoint_comparison.csv',dict(step=step,seed=1,samples=len(rows),source=str(source),sha256=common.sha(source),
            preprocessing='LEGACY_DOUBLE_NORM',policy_chunk_fps=1000/float(frame.inference_ms.mean()),
            latency_scope='policy chunk only; excludes camera/serial/preprocessing',
            **{k:float(frame[k].mean()) for k in ['arm_normalized','gripper_normalized','arm_raw','gripper_raw','h0_gripper_normalized','gate','object_on_off','inference_ms']}))
        print('LEGACY_EVAL',step,'DONE',flush=True)
        del policy;torch.cuda.empty_cache()


if __name__=='__main__':main()
