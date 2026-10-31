"""No-cup diagnostics: read observations and log predictions, never send actions."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--light',choices=['bright','dim'],required=True)
    p.add_argument('--position',choices=['A','B','C'],required=True)
    p.add_argument('--repeat',type=int,choices=[1,2,3],default=1)
    p.add_argument('--seed',type=int,default=4701)
    p.add_argument('--robot.port',dest='port',default='/dev/ttyACM0')
    p.add_argument('--robot.id',dest='robot_id',default='nn')
    p.add_argument('--top',type=int,default=4)
    p.add_argument('--gripper',type=int,default=2)
    args=p.parse_args();assert args.seconds>0
    root=args.bundle.resolve();sys.path.insert(0,str(root/'source/src'));sys.path.insert(0,str(root))
    import torch
    from p47_runtime import Runtime
    from lerobot.robots.so_follower import SO101Follower,SO101FollowerConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    torch.set_num_threads(4)
    identity=json.loads((root/'manifest.json').read_text());diffusion=identity['role']=='VDP-S1'
    runtime=Runtime(root,n_action_steps=None if diffusion else 1,ensemble=None if diffusion else .01)
    torch.manual_seed(args.seed)
    diagnostics=dict(object_branch_present=False);hook=None
    branch=getattr(runtime.policy.model,'object_conditioning',None) if not diffusion else None
    if branch is not None:
        def capture(_module,_inputs,outputs):
            tokens,similarity=outputs
            diagnostics.update(object_branch_present=True,global_xyc=tokens[0,0,-3:].detach().cpu().tolist(),
                               local_xyc=tokens[0,1:,-3:].detach().cpu().tolist(),
                               patch_similarity_max=float(similarity.max()),
                               patch_similarity_mean=float(similarity.mean()),patch_argmax=int(similarity[0].argmax()))
        hook=branch.tokenizer.register_forward_hook(capture)
    robot=SO101Follower(SO101FollowerConfig(port=args.port,id=args.robot_id,use_degrees=True,
        cameras={name:OpenCVCameraConfig(index_or_path=index,width=640,height=480,fps=30,fourcc='MJPG')
                 for name,index in [('top',args.top),('gripper',args.gripper)]}))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    output=args.output
    if output.exists():output=output.with_name(output.stem+'_'+str(time.time_ns())+output.suffix)
    try:
        robot.connect(calibrate=False);runtime.reset();start=time.perf_counter()
        with output.open('w') as stream:
            frame=0
            while time.perf_counter()-start<args.seconds:
                action,_,_=runtime.step(robot.get_observation())
                stream.write(json.dumps(dict(t=time.time(),role=identity['role'],light=args.light,position=args.position,
                    repeat=args.repeat,seed=args.seed,frame=frame,water='absent',executed=False,prediction=action.tolist(),**diagnostics))+'\n')
                frame+=1
    finally:
        if hook:hook.remove()
        runtime.reset();robot.disconnect()


if __name__=='__main__':main()
