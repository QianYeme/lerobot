"""Bounded isolated replay to capture the A0 nonfinite-gradient failure; never resume a formal run."""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch

from run_f2_1 import Experiment, shared_hash


def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [cpu(v) for v in value]
    return value


def stats(tensor):
    tensor=tensor.detach()
    finite=torch.isfinite(tensor)
    good=tensor[finite]
    return dict(shape=list(tensor.shape),nonfinite=int((~finite).sum()),
                min=float(good.min()) if good.numel() else None,max=float(good.max()) if good.numel() else None)


def gradient_audit(policy):
    gradients={name:p.grad for name,p in policy.named_parameters() if p.grad is not None}
    bad={name:stats(grad) for name,grad in gradients.items() if not torch.isfinite(grad).all()}
    if bad:
        return dict(all_gradient_elements_finite=False,bad_parameters=bad,n_parameters_with_grad=len(gradients))
    norm64=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.double()) for g in gradients.values()]))
    norm32=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in gradients.values()]))
    return dict(all_gradient_elements_finite=True,norm_fp64=float(norm64),
                norm_fp32=float(norm32) if torch.isfinite(norm32) else 'NONFINITE',n_parameters_with_grad=len(gradients))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--max-steps',type=int,default=15000);ap.add_argument('--replay-capture',type=Path)
    args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True
    exp=Experiment(args.root,args.data,args.out)
    args.out.mkdir(exist_ok=False)
    policy=exp.policy('A0')
    initial=shared_hash(policy)
    assert initial=='a7105e4b47051dc74e2afe5247a4a20c127f4147a159e82a0cc86b18e163baf5'
    if args.replay_capture:
        capture=torch.load(args.replay_capture,map_location='cpu',weights_only=False)
        policy.load_state_dict(capture['policy'],strict=True)
        batch={k:v.to('cuda') for k,v in capture['batch'].items()}
        torch.set_rng_state(capture['cpu_rng']);torch.cuda.set_rng_state(capture['cuda_rng'])
        try:
            with torch.autograd.detect_anomaly(check_nan=True):
                loss,parts=policy(batch);loss.backward()
            audit=gradient_audit(policy)
            report=dict(loss=stats(loss),parts=parts,gradients=audit,anomaly=None)
        except RuntimeError as error:
            traceback.print_exc()
            report=dict(anomaly=str(error),gradients=gradient_audit(policy))
        (args.out/'capture_replay.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print('[DEBUG-F21-A0] EXACT_CAPTURE',json.dumps(report),flush=True)
        return
    optimizer=torch.optim.AdamW(policy.get_optim_params(),lr=1e-5,weight_decay=1e-4)
    loader=exp.loader('train');dev=exp.loader('dev');iterator=iter(loader)
    outputs={}
    policy.model.register_forward_hook(lambda module,inputs,result:outputs.update(pred=result[0].detach(),
        mu=result[1][0].detach(),logvar=result[1][1].detach()))
    started=time.monotonic()
    for step in range(1,args.max_steps+1):
        if time.monotonic()-started>1800:
            print('[DEBUG-F21-A0] WALL_BUDGET_STOP',step,flush=True);break
        try:raw=next(iterator)
        except StopIteration:iterator=iter(loader);raw=next(iterator)
        policy.train();optimizer.zero_grad(set_to_none=True)
        batch=exp.batch(raw,'A0')
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
        loss,parts=policy(batch)
        failure=None
        if not torch.isfinite(loss):failure='NONFINITE_LOSS'
        else:
            loss.backward()
            try:grad_norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),10.,error_if_nonfinite=True)
            except RuntimeError as error:failure=str(error)
        if failure:
            report=dict(status='CAPTURED_NONFINITE_FAILURE',step=step,elapsed_s=time.monotonic()-started,
                failure=failure,loss=stats(loss),parts={k:v if abs(v)<1e300 else str(v) for k,v in parts.items()},
                input_stats={k:stats(v) for k,v in batch.items()},outputs={k:stats(v) for k,v in outputs.items()},
                gradients=gradient_audit(policy),episodes=raw['episode_index'].reshape(-1).tolist(),
                frames=raw['frame_index'].reshape(-1).tolist(),initial_hash=initial,
                scientific_use='DIAGNOSTIC_ONLY_NOT_A0_RESULT',formal_jobs_modified=False)
            torch.save(dict(policy=cpu(policy.state_dict()),optimizer=cpu(optimizer.state_dict()),batch=cpu(batch),
                cpu_rng=cpu_rng,cuda_rng=cuda_rng,step=step),args.out/'failure_capture.pt')
            (args.out/'failure_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
            print('[DEBUG-F21-A0] FAILURE',json.dumps(report),flush=True)
            return
        optimizer.step()
        if step%100==0 or step==1:
            row=dict(step=step,loss=float(loss.detach()),**parts,grad_norm=float(grad_norm),elapsed_s=time.monotonic()-started)
            print('[DEBUG-F21-A0] PROGRESS',json.dumps(row),flush=True)
            (args.out/'progress.json').write_text(json.dumps(row,indent=2))
    (args.out/'no_reproduction.json').write_text(json.dumps(dict(status='NOT_REPRODUCED_WITHIN_FIXED_BUDGET',
        completed_steps=step,elapsed_s=time.monotonic()-started,formal_jobs_modified=False),indent=2))


if __name__=='__main__':main()
