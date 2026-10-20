"""Audit completed F2.1 records and export evidence; never select outside frozen Dev checkpoints."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/force/f2_1_load_temporal_encoding_ablation/index1/result'
RAW=ROOT/'results/experiments/F2_1_LOAD_TEMPORAL_ENCODING_ABLATION/posttrain_20261019T1418'


def main():
    models={};curves=[];episodes=[];zero=[];perturb=[]
    dev=set(json.loads((OUT/'split_manifest.json').read_text())['dev'])
    for name,relative in [('A0','results_A0_retry1/A0'),('A1','results/A1'),('A2','results/A2')]:
        folder=RAW/relative
        selected=json.loads((folder/'selected.json').read_text())
        completed=json.loads((folder/'complete.json').read_text())
        assert completed['steps']==100000
        checkpoints={}
        for step in [20000,40000,60000,80000,100000]:
            data=json.loads((folder/'evaluation'/f'{step:06d}.json').read_text())
            assert data['frames']==5748 and set(map(int,data['per_episode']))==dev
            assert sum(r['frames'] for r in data['per_episode'].values())==5748
            weighted=sum(r['primary']*r['frames'] for r in data['per_episode'].values())/5748
            assert np.isclose(weighted,data['primary'],atol=1e-10)
            assert all(np.isfinite(data[k]) for k in ['primary','physical_gripper_mae','physical_arm_mae','normalized_all_l1'])
            checkpoints[step]=data
            curves.append(dict(variant=name,step=step,primary=data['primary']))
        assert selected['step']==min(checkpoints,key=lambda step:(checkpoints[step]['primary'],step))
        data=checkpoints[selected['step']]
        assert np.isclose(selected['primary'],data['primary'])
        training=list(csv.DictReader((folder/'training.csv').open()))
        assert len(training)==1001 and int(training[-1]['step'])==100000
        assert all(np.isfinite(float(r[k])) for r in training for k in ['loss','l1_loss','kld_loss','grad_norm'])
        models[name]=dict(selected_step=selected['step'],primary=data['primary'],physical_gripper_mae=data['physical_gripper_mae'],
            physical_arm_mae=data['physical_arm_mae'],normalized_all_l1=data['normalized_all_l1'],
            per_episode=data['per_episode'],checkpoint=selected['checkpoint'],training_seconds=completed['elapsed_s'],
            final_logged_loss=float(training[-1]['loss']),training_rows=len(training),mechanisms={},legal_horizons=data['legal_horizons'])
        for ep,r in data['per_episode'].items():episodes.append(dict(variant=name,episode=int(ep),frames=r['frames'],primary=r['primary']))
        for mode in ['zero','plus_0.1std','minus_0.1std']:
            records=[r for r in data['mechanisms'] if r['mode']==mode]
            if name=='A0':assert not records;continue
            n=sum(r['n'] for r in records);assert n==5748
            row=dict(variant=name,mode=mode,frames=n,normalized_gripper_change=sum(r['gripper'] for r in records)/n,
                     normalized_arm_change=sum(r['arm'] for r in records)/n)
            models[name]['mechanisms'][mode]=row
            (zero if mode=='zero' else perturb).append(row)
    comparisons={}
    for candidate,reference in [('A2','A1'),('A1','A0'),('A2','A0')]:
        a,b=models[candidate],models[reference]
        comparisons[candidate+'_vs_'+reference]=dict(primary_delta=a['primary']-b['primary'],
            percent_change=100*(a['primary']/b['primary']-1),
            episodes_better=sum(a['per_episode'][str(ep)]['primary']<b['per_episode'][str(ep)]['primary'] for ep in dev),
            episodes_total=12,physical_arm_percent_change=100*(a['physical_arm_mae']/b['physical_arm_mae']-1))
    audit=dict(status='PASS',scope='record audit, not new predictions',models=models,comparisons=comparisons)
    (OUT/'completed_evaluation_audit.json').write_text(json.dumps(audit,indent=2))
    for name,rows in [('checkpoint_curve.csv',curves),('per_episode_metrics.csv',episodes),
                      ('load_zero_ablation.csv',zero),('load_perturbation.csv',perturb)]:
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(models={k:{a:b for a,b in v.items() if a not in ['per_episode','legal_horizons']} for k,v in models.items()},comparisons=comparisons),indent=2))


if __name__=='__main__':main()
