"""Uniform 5k Dev/gradient/heatmap and 20-init native simulation probes."""
import argparse
import inspect
import json
from pathlib import Path
import shutil
import sys

from p410_worker import ROOT,R5,WORK,OUT,old,VALUES
P49=ROOT/'experiments/p4_final_p1_training/p4_9_unified_gate_init/index1'

def checkpoint(variant):
    if variant=='G0':return R5/'result/runs/fixed/checkpoints/005000/resume.pt'
    if variant=='G1e-3':return P49/'result/runs/LB_FIXED_GATEPLUS/checkpoints/005000/resume.pt'
    if variant=='NOOBJ':return R5/'result/runs/act/checkpoints/005000/resume.pt'
    return OUT/f'train/{variant}/checkpoints/005000/resume.pt'

def load(variant):
    import torch
    kind='act' if variant=='NOOBJ' else 'fixed'
    cfg,policy=old.make_policy(kind);path=checkpoint(variant)
    state=torch.load(path,map_location='cpu',weights_only=False);assert state['step']==5000
    policy.load_state_dict(state['model'],strict=True);policy.eval()
    del state
    return cfg,policy,dict(path=str(path),sha256=old.sha(path),step=5000)

def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',required=True);p.add_argument('--mode',choices=['dev','rollout'],required=True);a=p.parse_args()
    variant=a.variant
    sys.path.insert(0,str(P49/'workspace/scripts'))
    if a.mode=='dev':
        import p49_evaluate as evaluator
        import p49_runtime as runtime
        runtime.JOBS[variant]=('lb','act' if variant=='NOOBJ' else 'fixed',1,5000)
        evaluator.OUT=OUT/'evaluation'/variant
        def paired_load(_job,_historical):
            cfg,policy,r=load(variant)
            return old,cfg,policy,None,old.CachedData('dev').stats,r['path']
        evaluator.load=paired_load
        evaluator.stock_load=lambda:load(variant)
        evaluator.dev(variant,False,act=variant=='NOOBJ')
        if variant!='NOOBJ':
            import torch
            import csv
            sys.path.insert(0,str(R5.parent/'index6/workspace'))
            from p48_r6_probe import Trace,gradients
            _,policy,_=load(variant);trace=Trace(policy);records=[]
            for split in ['train','dev']:
                data=old.CachedData(split);batch=data.batch(range(8))
                for repeat in range(2):
                    rows=gradients(policy,trace,batch,'fixed',5000,split)
                    for record in rows:record.update(variant=variant,repeat=repeat)
                    records.extend(rows)
            trace.close()
            with (OUT/'evaluation'/variant/'gradients.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
            # Frozen bank references and independent stage observations: no coordinate edits.
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            lock=json.loads((R5.parent/'index6/result/input_lock.json').read_text())
            maps=OUT/'prototype_heatmaps'/variant;maps.mkdir(parents=True,exist_ok=True)
            trace=Trace(policy)
            from p48_r6_probe import predict
            for sample in lock['samples']:
                data=old.CachedData(sample['split']);batch=data.batch([sample['index']])
                predict(policy,trace,batch)
                sim=trace.data['similarity'][0].detach().cpu().numpy()
                image=batch['_oc_raw_top'][0].permute(1,2,0).cpu().numpy()
                fig,axes=plt.subplots(1,2,figsize=(7,3))
                axes[0].imshow(image);axes[1].imshow(sim);axes[0].set_title(sample['id']);axes[1].set_title('DINO similarity (not GT)')
                for ax in axes:ax.axis('off')
                fig.tight_layout();fig.savefig(maps/(sample['id']+'.png'));plt.close(fig)
            trace.close()
            # Group-wise parameter change from the same reconstructed seed1 initialization.
            _,initial=old.make_policy('fixed')
            with torch.no_grad():initial.model.object_conditioning.gate.fill_(VALUES.get(variant,.001 if variant=='G1e-3' else 0.))
            changes=[]
            for name,value in policy.named_parameters():
                if 'object_conditioning' in name and value.requires_grad:
                    start=dict(initial.named_parameters())[name]
                    changes.append(dict(parameter=name,delta_l2=float((value-start).norm()),initial_l2=float(start.norm())))
            old.write(OUT/'evaluation'/variant/'parameter_changes.json',changes)
            old.write(OUT/'evaluation'/variant/'reference_evidence_pointer.json',dict(path=str(R5.parent/'index6/result/input_lock.json'),sha256=old.sha(R5.parent/'index6/result/input_lock.json'),note='Frozen r6 sources; no new prototype or target annotation'))
    else:
        import p48_r5_evaluate as evaluator
        evaluator.load=lambda _kind:load(variant)
        dest=OUT/'rollouts'/variant;dest.mkdir(parents=True,exist_ok=True)
        contract=json.loads((R5/'result/timing_contract.json').read_text())
        contract['paired_initial_ids']=json.loads((OUT/'init_contract.json').read_text())['exploratory_init_ids']
        contract['episode_seeds']=[1]*20
        old.write(dest/'timing_contract.json',contract)
        shutil.copyfile(R5/'result/train35_statistics.pt',dest/'train35_statistics.pt')
        evaluator.OUT=dest
        source=inspect.getsource(evaluator.rollout).replace('assert len(rows)==10','assert len(rows)==20').replace('episodes=10','episodes=20').replace('/10,','/20,').replace('TEN_PAIRED','TWENTY_PAIRED')
        exec(compile(source,'frozen_r5_rollout_twenty_init_count','exec'),evaluator.__dict__)
        evaluator.rollout('act' if variant=='NOOBJ' else 'fixed')

if __name__=='__main__':main()
