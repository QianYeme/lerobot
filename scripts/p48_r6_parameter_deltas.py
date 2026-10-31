"""Read-only5k/10k object parameter differences; never save model tensors."""
import csv
import json
from pathlib import Path
import sys

ROOT=Path('/root/autodl-tmp/lerobot-main')
R5=ROOT/'experiments/p4_public_benchmark/p4_8_libero_pilot/index5'
OUT=R5.parent/'index6/result'
sys.path.insert(0,str(R5/'workspace/scripts'))


def main():
    import torch
    torch.set_num_threads(4);rows=[]
    for kind in ['legacy','fixed']:
        with (OUT/f'checkpoint_5k_10k_comparison_{kind}.csv').open() as f:names=sorted({r['parameter'] for r in csv.DictReader(f)})
        a=torch.load(R5/'result/runs'/kind/'checkpoints/005000/resume.pt',map_location='cpu',weights_only=False)
        b=torch.load(R5/'result/runs'/kind/'checkpoints/010000/resume.pt',map_location='cpu',weights_only=False)
        for name in names:
            old=a['model'][name].double();new=b['model'][name].double();delta=new-old
            rows.append(dict(model=kind,checkpoint_step='5000_to_10000',probe_id=name,parameter=name,
                elements=old.numel(),old_l2=float(old.norm()),new_l2=float(new.norm()),delta_l2=float(delta.norm()),
                relative_l2_change=float(delta.norm()/old.norm().clamp_min(1e-12)),rms_change=float(delta.square().mean().sqrt()),
                units='parameter tensor norms',measurement='existing checkpoints only',eval_dtype='storedfloat32; norm accumulatorfloat64'))
        del a,b
    with (OUT/'parameter_5k_10k_deltas.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    path=OUT/'gradients.csv'
    with path.open() as f:grad=list(csv.DictReader(f))
    with (OUT/'object_signal_scales.csv').open() as f:
        gate={(r['model'],r['checkpoint_step']):float(r['alpha']) for r in csv.DictReader(f)}
    for r in grad:
        actual=0. if r['checkpoint_step']=='0' else gate[(r['model'],r['checkpoint_step'])]
        forced=float(r['alpha'])!=actual
        r['alpha_origin']='temporary_differentiable_hook_leaf' if forced else 'actual_parameter'
        r['ood_probe']=forced and float(r['alpha'])!=0
        r['optimizer_updates']=0
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(grad[0]));w.writeheader();w.writerows(grad)
    print('READ_ONLY_PARAMETER_DELTAS_DONE',len(rows),flush=True)


if __name__=='__main__':main()
