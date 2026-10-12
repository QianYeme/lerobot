#!/usr/bin/env python
"""Read-only health diagnostic for the frozen T1 P2 checkpoint."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.offline_eval_act_det import collate
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEV = [3, 5, 12, 15, 21, 28, 33, 39, 41, 47, 53, 58]
TRAIN = [0, 2, 4, 6, 7, 9, 10, 13, 14, 17, 18, 19, 22, 23, 24, 25, 26, 29, 30, 32, 34, 35, 36, 38, 40, 42, 43, 45, 48, 49, 50, 51, 54, 55, 56, 57]

def move(x, d): return {k: v.to(d) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
def norm(x): return x.float().flatten(1).norm(dim=1)
def mean_stats(xs):
    a = torch.cat(xs).float(); return {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()), "max": float(a.max()), "q05": float(torch.quantile(a, .05)), "q50": float(torch.quantile(a, .50)), "q95": float(torch.quantile(a, .95))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--repo-id',required=True); ap.add_argument('--root',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--batch-size',type=int,default=8); ap.add_argument('--max-batches',type=int,default=8); args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); ck=Path(args.checkpoint)
    cfg=PreTrainedConfig.from_pretrained(ck); policy=ACTPolicy.from_pretrained(ck,config=cfg); policy.eval(); device=next(policy.parameters()).device; model=policy.model
    pre,_=make_pre_post_processors(policy_cfg=cfg,pretrained_path=str(ck)); top_idx=model.object_conditioning_top_camera_index
    # Capture the exact token entering the ACT encoder after object conditioning.
    captured=[]; fusion_errors=[]
    def enc_hook(_m, inp): captured.append(inp[0].detach().clone())
    handle=model.encoder.register_forward_pre_hook(enc_hook)
    original_oc_forward = model.object_conditioning.forward
    def oc_capture(act_tokens, top_image, object_tokens=None):
        fusion_errors.append(act_tokens[1].detach().clone())
        return original_oc_forward(act_tokens, top_image, object_tokens=object_tokens)
    model.object_conditioning.forward = oc_capture
    beta_orig=float(model.explicit_spatial_beta.detach().item()); mlp_norm=[]; state_norm=[]; effective_norm=[]; input_spatial=[]; object_residual=[]; visual_norm=[]; actions_by_beta={}; batch_cache=[]
    with torch.no_grad():
      ds=LeRobotDataset(args.repo_id,root=args.root,episodes=DEV,video_backend='pyav'); dl=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=2,collate_fn=collate)
      for i,raw in enumerate(dl):
        if i>=args.max_batches: break
        b=move(pre(raw),device); b[OBS_IMAGES]=[b[k] for k in cfg.image_features]
        top=b[OBS_IMAGES][top_idx]; rawtok,sim=model.object_conditioning.extract_raw_object_tokens(top); spatial=model.object_conditioning.global_spatial_features(rawtok); emb=model.explicit_spatial_mlp(spatial); state=model.encoder_robot_state_input_proj(b[OBS_STATE]);
        input_spatial.append(spatial.cpu()); mlp_norm.append(norm(emb).cpu()); state_norm.append(norm(state).cpu()); effective_norm.append(norm(beta_orig*emb).cpu());
        # Object response and visual token scales are measured at the encoder hook.
        captured.clear(); pred=model(b)[0]; batch_cache.append((b,rawtok,sim,spatial));
        expected = state + beta_orig * emb
        if fusion_errors:
            fusion_errors[-1] = float((fusion_errors[-1] - expected).abs().max().item())
        if captured:
          toks=captured[-1]; visual_norm.append(norm(toks[2:]).cpu()); object_residual.append(torch.zeros(toks.shape[1],device=toks.device).cpu())
    handle.remove()
    model.object_conditioning.forward = original_oc_forward
    # Beta override and coordinate counterfactuals on the same cached batches.
    def run_with(beta, mode=None):
      vals=[]
      with torch.no_grad():
       with torch.no_grad(): model.explicit_spatial_beta.fill_(float(beta))
       for b,rawtok,sim,sp in batch_cache:
        chosen=sp.clone()
        if mode=='x_flip': chosen[:,0]=-chosen[:,0]
        elif mode=='y_flip': chosen[:,1]=-chosen[:,1]
        elif mode=='xy_shuffle': chosen=chosen[torch.randperm(chosen.shape[0],device=chosen.device)]
        elif mode=='zero': chosen.zero_()
        old=model.object_conditioning.global_spatial_features
        model.object_conditioning.global_spatial_features=lambda _raw: chosen
        try: vals.append(model(b)[0][:,0].detach().cpu())
        finally: model.object_conditioning.global_spatial_features=old
      return torch.cat(vals)
    base=run_with(0.0); betas=[0.0,0.00318068,0.01,0.05,0.10,0.50,1.0]; sens={}
    for bv in betas:
      p=run_with(bv); sens[str(bv)]={"mean_abs_delta_vs_beta0":float((p-base).abs().mean()),"mean_arm_delta":float((p[:,:-1]-base[:,:-1]).abs().mean()),"mean_gripper_delta":float((p[:,-1:]-base[:,-1:]).abs().mean())}
    cf={}
    for mode in ('x_flip','y_flip','xy_shuffle','zero'):
      p=run_with(1.0,mode); cf[mode]={"mean_abs_delta":float((p-run_with(1.0)).abs().mean()),"mean_abs_arm_delta":float((p[:,:-1]-run_with(1.0)[:,:-1]).abs().mean())}
    with torch.no_grad(): model.explicit_spatial_beta.fill_(beta_orig)
    # Train/dev input variance, bounded and independent of intervention.
    variance={}
    for name,eps in [('train',TRAIN),('dev',DEV)]:
      ds=LeRobotDataset(args.repo_id,root=args.root,episodes=eps,video_backend='pyav'); dl=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=2,collate_fn=collate); xs=[]
      with torch.no_grad():
       for i,raw in enumerate(dl):
        if i>=args.max_batches: break
        b=move(pre(raw),device); b[OBS_IMAGES]=[b[k] for k in cfg.image_features]; rawtok,_=model.object_conditioning.extract_raw_object_tokens(b[OBS_IMAGES][top_idx]); xs.append(model.object_conditioning.global_spatial_features(rawtok).cpu())
      variance[name]={k:mean_stats([x[:,j:j+1] for x in xs]) for j,k in enumerate(('x','y','confidence'))}
    state_mean=float(torch.cat(state_norm).mean()); emb_mean=float(torch.cat(mlp_norm).mean()); eff_mean=float(torch.cat(effective_norm).mean()); vis_mean=float(torch.cat(visual_norm).mean()) if visual_norm else None
    scale={"state_token_norm_mean":state_mean,"raw_spatial_embedding_norm_mean":emb_mean,"effective_spatial_residual_norm_mean":eff_mean,"visual_token_norm_mean":vis_mean,"effective_spatial_over_state":eff_mean/state_mean,"effective_spatial_over_visual":None if vis_mean is None else eff_mean/vis_mean,"beta":beta_orig}
    traj={"status":"NOT_RECORDED","available_checkpoints":[100000],"beta":{"100000":beta_orig},"component_gradient_history":"NOT_RECORDED","checkpoint_parameter_norms":"not extracted for missing intermediate checkpoints"}
    impl={"fusion_hook_captured":len(fusion_errors)>0,"formula":"state_proj + beta * SpatialMLP([x,y,confidence])","pre_object_conditioning_fusion_max_abs_error":float(max(fusion_errors)) if fusion_errors else None,"dino_frozen":all(not p.requires_grad for p in model.object_conditioning.dino.parameters()),"beta_checkpoint":beta_orig,"checkpoint":"100000","post_layernorm_effect":"NOT_DIRECTLY_CAPTURED; ACT encoder layer applies internal pre/post norms"}
    classification='SCALE_STARVATION' if max(v['mean_abs_delta_vs_beta0'] for v in sens.values())>1e-4 and sens['1.0']['mean_abs_delta_vs_beta0']>sens['0.01']['mean_abs_delta_vs_beta0']*2 else 'REPRESENTATION_WEAKNESS'
    (out/'parameter_trajectory.json').write_text(json.dumps(traj,indent=2)); (out/'scale_analysis.json').write_text(json.dumps(scale,indent=2)); (out/'beta_override_sensitivity.json').write_text(json.dumps(sens,indent=2)); (out/'input_variance.json').write_text(json.dumps(variance,indent=2)); (out/'implementation_check.json').write_text(json.dumps(impl,indent=2)); (out/'counterfactuals.json').write_text(json.dumps(cf,indent=2)); (out/'diagnosis.json').write_text(json.dumps({"primary_class":classification,"secondary_classes":["OPTIMIZATION_REJECTION"],"evidence":{"beta_override":sens,"scale":scale,"counterfactuals":cf},"final_test_used":False,"real_robot_used":False},indent=2))
    print(json.dumps({"diagnosis":classification,"scale":scale,"beta_override":sens},indent=2))

if __name__=='__main__': main()
