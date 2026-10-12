#!/usr/bin/env python
"""H0 read-only horizontal-control signal audit on frozen Formal4 splits."""
from __future__ import annotations

import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.offline_eval_act_det import collate
from lerobot.utils.constants import OBS_IMAGES

TRAIN = [0,2,4,6,7,9,10,13,14,17,18,19,22,23,24,25,26,29,30,32,34,35,36,38,40,42,43,45,48,49,50,51,54,55,56,57]
DEV = [3,5,12,15,21,28,33,39,41,47,53,58]
HORIZONS = [1, 3, 5, 10]


def rank_corr(a, b):
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12: return 0.0
    return float(np.corrcoef(pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy())[0, 1])


def bootstrap_ci(values, seed=17, n=2000):
    values = np.asarray(values, float)
    if len(values) < 3: return [None, None]
    rng = np.random.default_rng(seed); means = [np.mean(values[rng.integers(0, len(values), len(values))]) for _ in range(n)]
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def direction_stats(target, pred):
    target, pred = np.asarray(target), np.asarray(pred)
    mask = (np.abs(target) > 1e-8) & (np.abs(pred) > 1e-8)
    signs = (np.sign(target[mask]) == np.sign(pred[mask])) if mask.any() else np.array([])
    return {"n": int(len(target)), "n_nonzero": int(mask.sum()), "sign_accuracy": float(signs.mean()) if len(signs) else 0.0,
            "pearson": float(np.corrcoef(target, pred)[0,1]) if len(target)>2 and np.std(target)>1e-12 and np.std(pred)>1e-12 else 0.0,
            "spearman": rank_corr(target, pred), "mean_abs_pred": float(np.mean(np.abs(pred))) if len(pred) else 0.0,
            "sign_bootstrap_ci": bootstrap_ci(signs.astype(float)) if len(signs) else [None,None]}


def phase_rows(actions, states, episode_index, episodes):
    rows=[]; manifest={}
    for ep in episodes:
        ix=np.flatnonzero(episode_index==ep); a=actions[ix]; s=states[ix]; g=a[:,-1]
        lo=float(np.quantile(g[:max(10,len(g)//20)], .5)); hi=float(np.quantile(g, .95)); threshold=lo+0.20*(hi-lo)
        close=np.flatnonzero(g >= threshold)
        close_start=int(close[0]) if len(close) else int(len(g)*.35)
        # Reach/fine-alignment only: strictly before the first sustained closing excursion.
        sustain=5
        for j in range(max(0,close_start-8), len(g)-sustain+1):
            if np.all(g[j:j+sustain] >= threshold): close_start=j; break
        keep=np.arange(0, max(1,close_start))
        manifest[str(int(ep))]={"length":int(len(g)),"open_median":lo,"close_threshold":threshold,"close_start_frame":close_start,"selected_frames":int(len(keep)),"rule":"frames before first 5-frame sustained gripper closing threshold"}
        for j in keep: rows.append((int(ep),int(j),int(ix[j])))
    return rows, manifest


def ridge(X,y,lam=1e-3):
    mu=X.mean(0); sd=X.std(0); sd[sd<1e-8]=1.0; z=(X-mu)/sd
    A=z.T@z + lam*np.eye(z.shape[1]); w=np.linalg.solve(A,z.T@(y-y.mean()))
    return {"mean":mu,"std":sd,"weight":w,"intercept":float(y.mean()),"lambda":lam}


def predict(m, X): return m["intercept"] + ((X-m["mean"])/m["std"]) @ m["weight"]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--repo-id',required=True); ap.add_argument('--root',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--num-workers',type=int,default=4); args=ap.parse_args()
    out=Path(args.output_dir); (out/'figures').mkdir(parents=True,exist_ok=True)
    ds=LeRobotDataset(args.repo_id,root=args.root,episodes=TRAIN+DEV,video_backend='pyav')
    parquet=Path(args.root)/'data/chunk-000/file-000.parquet'; table=pd.read_parquet(parquet); actions=np.stack(table.action.to_numpy()); states=np.stack(table['observation.state'].to_numpy()); episodes_all=table.episode_index.to_numpy(); frames_all=table.frame_index.to_numpy()
    phase_manifest={}; train_rows,tm=phase_rows(actions,states,episodes_all,TRAIN); dev_rows,dm=phase_rows(actions,states,episodes_all,DEV); phase_manifest.update(tm); phase_manifest.update(dm)
    (out/'phase_selection_manifest.json').write_text(json.dumps({"train_episodes":TRAIN,"dev_episodes":DEV,"final_test_used":False,"rule":"reach/fine-alignment frames before first sustained gripper closing threshold","episodes":phase_manifest},indent=2))
    # Frozen P0.1 spatial extraction from the OC-ACT object branch.
    cfg=PreTrainedConfig.from_pretrained(args.checkpoint); policy=ACTPolicy.from_pretrained(args.checkpoint,config=cfg); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); policy.to(device); policy.eval()
    pre,_=make_pre_post_processors(policy_cfg=cfg,pretrained_path=str(args.checkpoint)); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,collate_fn=collate)
    spatial={}; oc=policy.model.object_conditioning; top_idx=policy.model.object_conditioning_top_camera_index
    with torch.no_grad():
        for raw in loader:
            ep=raw['episode_index'].numpy().reshape(-1); fr=raw['frame_index'].numpy().reshape(-1); b=pre(raw); b={k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in b.items()}; b[OBS_IMAGES]=[b[k] for k in cfg.image_features]; top=b[OBS_IMAGES][top_idx]; rawtok,sim=oc.extract_raw_object_tokens(top); sp=oc.global_spatial_features(rawtok).detach().cpu().numpy(); conf=sim.detach().cpu().numpy().reshape(-1)
            for e,f,z,c in zip(ep,fr,sp,conf): spatial[(int(e),int(f))]=[float(z[0]),float(z[1]),float(z[2]),float(c)]
    (out/'coordinate_convention.json').write_text(json.dumps({"image_x":"pixel column increasing right","target_x":"P0.1 x_global; increasing image-coordinate direction","robot_horizontal_sign":"not independently calibrated; empirical Method-B projection sign is frozen from Train","fk_jacobian":"unavailable in project; Method B ridge projection used","final_test_used":False},indent=2))
    def build(rows):
        rec=[]
        for ep,fr,ix in rows:
            if (ep,fr) not in spatial: continue
            rec.append({"episode":ep,"frame":fr,"global_index":ix,"x":spatial[(ep,fr)][0],"y":spatial[(ep,fr)][1],"confidence":spatial[(ep,fr)][3]})
        return rec
    tr=build(train_rows); dv=build(dev_rows); np.savez(out/'h0_arrays.npz', train_indices=np.array([r['global_index'] for r in tr]), dev_indices=np.array([r['global_index'] for r in dv]))
    # Build frozen action features for each reach row and horizon.
    by_ep={int(e):np.flatnonzero(episodes_all==e) for e in set(TRAIN+DEV)}
    def features(records):
        result={h:[] for h in HORIZONS}
        for r in records:
            ep,fr,ix=r['episode'],r['frame'],r['global_index']; epix=by_ep[ep]; pos=np.where(epix==ix)[0][0]
            for h in HORIZONS:
                j=min(pos+h-1,len(epix)-1); result[h].append(actions[epix[j],:5]-states[ix,:5])
        return {h:np.asarray(v,float) for h,v in result.items()}
    Xtr=features(tr); Xdv=features(dv); ytr=np.array([r['x'] for r in tr]); ydv=np.array([r['x'] for r in dv])
    mapping={}; temporal={}
    for h in HORIZONS:
        m=ridge(Xtr[h],ytr); mapping[str(h)]={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in m.items()}; p=predict(m,Xtr[h]); q=predict(m,Xdv[h]); temporal[str(h)]={"train":direction_stats(ytr-p*0 + (ytr-ytr.mean()),p-p.mean()),"dev":direction_stats(ydv-ydv.mean(),q-q.mean()),"weights":mapping[str(h)]}
    (out/'horizontal_mapping.json').write_text(json.dumps({"method":"B_train_only_ridge","features":"5 arm joint target-action minus current state","target":"x_global","horizons":HORIZONS,"mapping":mapping,"train_count":len(tr),"dev_count":len(dv)},indent=2))
    (out/'expert_joint_signal.json').write_text(json.dumps({"train_count":len(tr),"dev_count":len(dv),"action_dimensions":["shoulder_pan.pos","shoulder_lift.pos","elbow_flex.pos","wrist_flex.pos","wrist_roll.pos"],"target":"x_global","horizons":temporal,"secondary_dim0_only":True},indent=2))
    # State-matched pairs: fixed nearest-neighbour rule, selected on Train only.
    def pair_records(records):
        outp=[]; used=set(); ep_states=[]
        for r in records:
            ix=r['global_index']; ep_states.append((r,np.r_[states[ix,:5], r['frame']/479.0]))
        if not ep_states:return outp
        Z=np.asarray([z for _,z in ep_states]); scale=np.std(Z,0); scale[scale<1e-8]=1; Z=Z/scale
        for i,(r,z) in enumerate(zip([x[0] for x in ep_states],Z)):
            cand=[]
            for j,(rr,zz) in enumerate(zip([x[0] for x in ep_states],Z)):
                if j==i or j in used or rr['episode']==r['episode'] or abs(rr['frame']/479-r['frame']/479)>.20: continue
                dx=abs(rr['x']-r['x'])
                if dx >= .10: cand.append((np.linalg.norm(z-zz),j,rr))
            if cand:
                _,j,rr=min(cand); used|={i,j}; outp.append({"left_or_base":r,"matched":rr,"target_dx":rr['x']-r['x']})
            if len(outp)>=100: break
        return outp
    pairs=pair_records(tr); dev_pairs=pair_records(dv); (out/'state_matched_pairs.json').write_text(json.dumps({"train_pairs":pairs,"dev_pairs":dev_pairs,"min_abs_dx":.10,"phase_rule":"same reach phase and normalized time tolerance <=0.20","count_train":len(pairs),"count_dev":len(dev_pairs)},indent=2))
    # Validate frozen mapping on pair actions.
    m=ridge(Xtr[1],ytr)
    def pair_eval(ps,records,X):
        lookup={r['global_index']:i for i,r in enumerate(records)}; tx=[]; pp=[]
        for p in ps:
            a,b=p['left_or_base'],p['matched'];
            if a['global_index'] in lookup and b['global_index'] in lookup: tx.append(b['x']-a['x']); pp.append(predict(m,X[lookup[b['global_index']]])-predict(m,X[lookup[a['global_index']]]))
        return direction_stats(np.asarray(tx),np.asarray(pp))
    (out/'dev_mapping_validation.json').write_text(json.dumps({"mapping_horizon":1,"train_pairs":pair_eval(pairs,tr,Xtr[1]),"dev_pairs":pair_eval(dev_pairs,dv,Xdv[1]),"gate":"mapping is diagnostic; no model selection performed"},indent=2))
    (out/'expert_temporal_signal.json').write_text(json.dumps({"horizons":temporal,"selection_frozen":True,"target":"x_global","projected_horizontal":"ridge(q_delta)"},indent=2))
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,4)); plt.scatter(ytr,predict(m,Xtr[1]),s=3,alpha=.25); plt.xlabel('cup x_global'); plt.ylabel('projected horizontal'); plt.tight_layout(); plt.savefig(out/'figures/cup_x_vs_projected_horizontal.png',dpi=160); plt.close()
        plt.figure(figsize=(6,4)); plt.bar([str(h) for h in HORIZONS],[temporal[str(h)]['train']['dev'] if False else temporal[str(h)]['train']['mean_abs_pred'] for h in HORIZONS]); plt.xlabel('horizon'); plt.ylabel('signal magnitude'); plt.tight_layout(); plt.savefig(out/'figures/horizon_signal_strength.png',dpi=160); plt.close()
    except Exception: pass
    validation={"experiment":"H0_HORIZONTAL_CONTROL_SIGNAL_AUDIT","train_episodes":TRAIN,"dev_episodes":DEV,"final_test_used":False,"policy_training":False,"architecture_change":False,"real_robot_used":False,"fk_available":False,"method":"B_train_only_empirical_projection","status":"complete"}
    (out/'validation.json').write_text(json.dumps(validation,indent=2)); print(json.dumps({"train_frames":len(tr),"dev_frames":len(dv),"train_pairs":len(pairs),"dev_pairs":len(dev_pairs),"device":str(device)},indent=2))


if __name__=='__main__': main()
