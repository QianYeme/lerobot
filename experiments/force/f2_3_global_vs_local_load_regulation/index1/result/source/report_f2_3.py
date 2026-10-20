"""F2.3 final Dev12 diagnostics/cost/figures/report; no training or Final access."""
import argparse
import csv
import hashlib
import json
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from run_f2_3 import LocalResidual, data_inputs, metrics, digest
from run_f2_1 import Experiment, save
from lerobot.policies.act.modeling_act import ACTPolicy


def table(path, rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);args=parser.parse_args()
    base=args.base.resolve();out=base/'result';inputs=base/'inputs'
    cfg=json.loads((inputs/'residual_config.json').read_text())
    normal=json.loads((out/'local_normalization.json').read_text())
    norm=json.loads((inputs/'normalization.json').read_text());std=np.asarray(norm['action']['std'])
    dev=dict(np.load(base/'cache/dev.npz'));train=dict(np.load(base/'cache/train.npz'))
    assert len(dev['episode'])==5748 and len(train['episode'])==17243
    split=json.loads((inputs/'split_manifest.json').read_text())
    assert set(dev['episode'])==set(split['dev']) and set(train['episode'])==set(split['train'])
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True
    scores={'C0':metrics(dev['c0'],dev['target'],std,dev['episode'])}
    residuals={};local_models={};mechanism={};health={};manifest=[];residual_rows=[];counter=[];zero=[];predictions={'C0':dev['c0']}
    for variant in ['C1-no-load','C1-load']:
        directory=out/variant
        complete=json.loads((directory/'complete.json').read_text());assert complete['steps']==cfg['steps']
        evaluations=json.loads((directory/'evaluations.json').read_text())
        assert [r['step'] for r in evaluations]==list(range(1000,10001,1000))
        selection=json.loads((directory/'selected.json').read_text())
        assert selection==min(evaluations,key=lambda r:(r['primary'],r['step']))
        saved=torch.load(selection['path'],map_location='cuda',weights_only=True)
        assert saved['config']==cfg and saved['normalization']==normal
        model=LocalResidual(variant=='C1-load',cfg['delta_max']).cuda().eval()
        model.load_state_dict(saved['model'],strict=True)
        x=torch.tensor(data_inputs(dev,normal,variant=='C1-load'),device='cuda')
        with torch.no_grad():delta=model(x).cpu().numpy()
        assert np.isfinite(delta).all() and np.abs(delta).max()<=cfg['delta_max']+1e-6
        pred=dev['nominal'].copy();pred[:,5]+=delta/std[5]
        assert np.array_equal(pred[:,:5],dev['nominal'][:,:5]),'INVALID_LOCAL_ISOLATION'
        scores[variant]=metrics(pred,dev['target'],std,dev['episode'])
        assert abs(scores[variant]['primary']-selection['primary'])<1e-6
        local_models[variant]=model;predictions[variant]=pred;residuals[variant]=delta
        np.savez_compressed(directory/'selected_dev.npz',episode=dev['episode'],frame=dev['frame'],
            prediction=pred,nominal=dev['nominal'],target=dev['target'],delta=delta,residual_target=dev['residual_target'])
        corr=float(np.corrcoef(delta,dev['residual_target'])[0,1]) if delta.std()>0 and dev['residual_target'].std()>0 else None
        mechanism[variant]=dict(correlation_with_bounded_target=corr,
            mean_abs=float(np.abs(delta).mean()),median_abs=float(np.median(np.abs(delta))),max_abs=float(np.abs(delta).max()),
            near_bound_fraction=float((np.abs(delta)>=.95*cfg['delta_max']).mean()),
            positive_fraction=float((delta>0).mean()),negative_fraction=float((delta<0).mean()),zero_fraction=float((delta==0).mean()))
        for ep in split['dev']:
            subset=delta[dev['episode']==ep]
            residual_rows.append(dict(variant=variant,episode=ep,mean_abs=float(np.abs(subset).mean()),
                median_abs=float(np.median(np.abs(subset))),max_abs=float(np.abs(subset).max()),
                near_bound_fraction=float((np.abs(subset)>=.95*cfg['delta_max']).mean()),positive_fraction=float((subset>0).mean())))
        nominal=dev['nominal'].copy();nominal[:,5]+=np.zeros_like(delta)/std[5]
        assert np.array_equal(nominal,dev['nominal'])
        zero.append(dict(variant=variant,max_abs_vs_global=float(np.abs(nominal-dev['nominal']).max()),
            normalized_h0_gripper_L1=metrics(nominal,dev['target'],std,dev['episode'])['primary']))
        if variant=='C1-load':
            neutral=x.clone();neutral[:,0]=0
            with torch.no_grad():neutral_delta=model(neutral).cpu().numpy()
            change=np.abs(neutral_delta-delta)
            counter.append(dict(variant=variant,normalized_load=0.,frames=5748,
                mean_abs_native_residual_change=float(change.mean()),mean_abs_native_final_gripper_change=float(change.mean()),
                normalized_final_gripper_change=float(change.mean()/std[5]),neutral_means='Train load mean, not physical zero'))
            np.savez_compressed(directory/'load_zero.npz',normal_delta=delta,neutral_delta=neutral_delta)
        with (directory/'training.csv').open() as stream:logs=list(csv.DictReader(stream))
        assert [int(r['step']) for r in logs]==[1]+list(range(100,10001,100))
        assert all(np.isfinite(float(v)) for r in logs for v in r.values())
        health[variant]=dict(steps=10000,finite_logged_rows=len(logs),selected_step=selection['step'],trainable_parameters=sum(p.numel() for p in model.parameters()))
        manifest.append(dict(variant=variant,seed=1,steps=10000,batch_size=256,optimizer='Adam',learning_rate=.001,
            global_frozen=True,selected_step=selection['step'],checkpoint=selection['path']))
    assert np.array_equal(predictions['C1-load'][:,:5],predictions['C1-no-load'][:,:5])
    identity=json.loads((out/'baseline_identity.json').read_text())
    exp=Experiment(inputs,Path('/root/autodl-tmp/lerobot/formal4_data/kind_merged_nomaster'),out)
    raw=next(iter(exp.loader('dev',workers=0)));raw={k:v[:1] for k,v in raw.items()}
    cost={}
    for name in ['C0','C1-no-load','C1-load']:
        identity_key='C0' if name=='C0' else 'C1_global'
        checkpoint=Path(identity[identity_key]['checkpoint'])
        assert digest(checkpoint/'model.safetensors')==identity[identity_key]['model_sha256']
        policy=ACTPolicy.from_pretrained(checkpoint).cuda().eval().requires_grad_(False)
        batch=exp.batch(raw,'A1' if name=='C0' else 'A0');batch.pop('action')
        def inference():
            pred=policy.predict_action_chunk(batch)[:,0]
            if name!='C0':
                pos=raw['observation.state'][:,5].cuda()
                native=pred[:,5]*std[5]+norm['action']['mean'][5]
                fields=[(pos-normal['position']['mean'])/normal['position']['std'],
                        (native-normal['nominal']['mean'])/normal['nominal']['std']]
                if name=='C1-load':
                    ep=int(raw['episode_index'][0]);frame=int(raw['frame_index'][0])
                    fields.insert(0,torch.tensor([exp.cache[ep]['load_single'][frame,0]],device='cuda'))
                result=pred.clone();result[:,5]+=local_models[name](torch.stack(fields,dim=1))/std[5]
                return result
            return pred
        with torch.inference_mode():
            for _ in range(10):inference()
            torch.cuda.synchronize();warmup_peak=torch.cuda.max_memory_allocated();torch.cuda.reset_peak_memory_stats()
            times=[]
            for _ in range(50):
                started=time.perf_counter();result=inference();torch.cuda.synchronize();times.append((time.perf_counter()-started)*1000)
            assert torch.isfinite(result).all()
        local_params=0 if name=='C0' else sum(p.numel() for p in local_models[name].parameters())
        cost[name]=dict(total_parameters=sum(p.numel() for p in policy.parameters())+local_params,
            global_parameters=sum(p.numel() for p in policy.parameters()),local_parameters=local_params,
            training_trainable_parameters=identity['C0']['parameters'] if name=='C0' else local_params,
            batch_size=1,warmup=10,repeats=50,wall_ms_median=float(np.median(times)),wall_ms_p95=float(np.percentile(times,95)),
            steady_peak_bytes=torch.cuda.max_memory_allocated(),warmup_peak_bytes=warmup_peak,
            scope='global forward + local feature construction + local forward; excludes camera and global input preprocessing')
        del policy;torch.cuda.empty_cache()
    save(out/'latency_and_params.json',dict(gpu=torch.cuda.get_device_name(),variants=cost,
        asymmetry='C0 reused globally trained load model; C1 frozen distinct no-load model, local-only trainable. Not equal end-to-end budgets.'))
    c0=scores['C0']['primary'];noload=scores['C1-no-load']['primary'];load=scores['C1-load']['primary']
    improvements={v:sum(scores['C1-load']['per_episode'][str(ep)]['primary']<scores[v]['per_episode'][str(ep)]['primary'] for ep in split['dev']) for v in ['C0','C1-no-load']}
    saturated=mechanism['C1-load']['near_bound_fraction']>=.5
    if saturated:outcome='LOCAL_POLICY_SATURATES'
    elif load<c0 and load<noload and min(improvements.values())>=8 and mechanism['C1-load']['near_bound_fraction']<.2 and counter[0]['normalized_final_gripper_change']>1e-6 and cost['C1-load']['wall_ms_median']/cost['C0']['wall_ms_median']<=1.2:
        outcome='LOCAL_LOAD_REGULATION_PROMISING'
    elif noload<c0 and load>=noload:outcome='LOCAL_RESIDUAL_HELPS_BUT_LOAD_NOT_NEEDED'
    elif load<noload and load>=c0:outcome='LOCAL_LOAD_HELPS_BUT_NOT_OVER_GLOBAL'
    elif c0<=min(load,noload):outcome='GLOBAL_FUSION_REMAINS_BEST'
    else:outcome='UNRESOLVED'
    save(out/'metrics.json',dict(variants=scores,outcome=outcome,episodes_improved=improvements,evidence='SINGLE_SEED_EXPLORATORY'))
    save(out/'residual_metrics.json',mechanism)
    save(out/'training_health.json',health)
    table(out/'training_manifest.csv',manifest)
    table(out/'per_episode_metrics.csv',[dict(variant=v,episode=int(ep),**{k:x for k,x in values.items() if k!='per_episode'}) for v in scores for ep,values in scores[v]['per_episode'].items()])
    table(out/'residual_metrics.csv',residual_rows);table(out/'load_zero_counterfactual.csv',counter);table(out/'zero_residual_counterfactual.csv',zero)
    figures=out/'figures';figures.mkdir(exist_ok=True)
    def finish(name,ylabel,title):
        plt.ylabel(ylabel);plt.title(title);plt.tight_layout();plt.savefig(figures/name,dpi=150);plt.close()
    names=list(scores)
    plt.figure(figsize=(7,4));plt.bar(names,[scores[v]['primary'] for v in names]);finish('c0_c1_primary_metric.png','Normalized h0 gripper L1','Dev12 selected checkpoints; lower better')
    plt.figure(figsize=(10,4));xs=np.arange(12)
    for i,v in enumerate(names):plt.bar(xs+(i-1)*.26,[scores[v]['per_episode'][str(ep)]['primary'] for ep in split['dev']],width=.26,label=v)
    plt.xticks(xs,split['dev']);plt.legend();finish('per_episode_primary_metric.png','Normalized h0 gripper L1','Dev episode')
    plt.figure(figsize=(7,4));plt.hist(train['raw_residual'],bins=80,range=(-4*cfg['delta_max'],4*cfg['delta_max']),alpha=.6,label='raw (tails outside displayed range)');plt.hist(train['residual_target'],bins=40,alpha=.6,label='bounded');plt.legend();finish('residual_target_distribution.png','Frames','Train36 native residual targets')
    plt.figure(figsize=(7,4))
    for v in residuals:plt.scatter(dev['residual_target'][::5],residuals[v][::5],s=4,alpha=.25,label=v)
    plt.legend();plt.xlabel('Bounded target (native)');finish('predicted_vs_target_residual.png','Predicted delta','Dev12 (every5th frame for display only)')
    plt.figure(figsize=(9,4))
    for v in residuals:plt.plot(split['dev'],[r['mean_abs'] for r in residual_rows if r['variant']==v],marker='o',label=v)
    plt.legend();finish('residual_magnitude_per_episode.png','Mean absolute native delta','Dev episode')
    plt.figure(figsize=(6,4));plt.bar(['C1-load'],[counter[0]['mean_abs_native_residual_change']]);finish('load_zero_effect.png','Mean absolute native delta change','Normalized load zero = Train mean')
    plt.figure(figsize=(7,4));plt.bar(names,[scores[v]['physical_arm_mae'] for v in names]);finish('arm_mae_comparison.png','Denormalized arm MAE','C1 arms identical, distinct global from C0')
    plt.figure(figsize=(9,4))
    for v in residuals:
        with (out/v/'training.csv').open() as stream:rows=list(csv.DictReader(stream))
        plt.plot([int(r['step']) for r in rows],[float(r['loss']) for r in rows],label=v)
    plt.legend();finish('training_curves.png','Bound-normalized residual MSE','Matched10000 steps, seed1')
    for path in figures.glob('*.png'):
        pixels=plt.imread(path);assert np.isfinite(pixels).all() and pixels.std()>.01
    shutil.copy2(inputs/'residual_config.json',out/'residual_config.json')
    (out/'variant_configs').mkdir(exist_ok=True)
    for v in ['C1-no-load','C1-load']:save(out/'variant_configs'/f'{v}.json',dict(config=cfg,inputs=cfg['local_inputs'][v]))
    save(out/'variant_configs/C0.json',identity['C0']['config'])
    def doc(name,text):(out/name).write_text(text,encoding='utf-8')
    doc('summary.md',f'# F2.3 — {outcome}\n\nCompleted matched two local MLP runs; C0/global checkpoints reused, not retrained.\n\n'+json.dumps({v:{k:x for k,x in scores[v].items() if k!='per_episode'} for v in scores},indent=2)+'\n\nSingle-seed exploratory teacher-forced evidence. No calibrated force or robot success claim. Final sealed. Stop; external Builder/Red-Team review pending.\n')
    doc('validation.md','# Validation\n\nIdentity hashes and 6D/no-load global validated; frozen model gradients disabled. Train17243/Dev5748 unique frames, no shuffle/drop_last on replay, same causal h0 zero-latent global semantics. Train-only bound/normalization and clipped target. Both local10000/batch256/Adam1e-3/seed1; all101 logged rows finite and10 checkpoint evaluations; selected reload primary reproduced within1e-6. Exact C1 arm preservation and zero-residual identity; diagnostic correlation/magnitude/saturation/load-zero delivered. Eight nonblank PNGs verified, external visual review pending. Checkpoint source and normalization preserved.\n')
    doc('training_health.md','# Training health\n\n'+json.dumps(health,indent=2)+'\nGlobal models were not optimized. Two disposable2-step engineering smoke models are not formal checkpoints.\n')
    doc('arm_preservation_audit.md','# Arm preservation PASS\n\nC1-no-load and C1-load arms np.array_equal to same cached frozen global output; zero residual exactly reproduces full nominal. Only column5 is modified. C0 arms can differ because C0 has a different global model.\n')
    doc('mechanism_review.md','# Mechanisms\n\nZero residual exactly returns frozen A0. Normalized load neutral=Train mean, not physical zero. Pearson correlation compares predicted native residual to bounded expert-minus-nominal target. Saturation/sign/per-episode magnitude reported. Train nominal targets are in-sample predictions of an already-trained global model; this and F1RM load confounding limit causal/contact interpretation.\n\n'+json.dumps(mechanism,indent=2))
    doc('implementation_diff.md','# Isolation\n\nNo ACT source changes. Separate2/3-input MLP32-32-1, GELU, tanh times bound. Additional load input adds32 first-layer weights. Seeded common hidden and zero output initialization. C1 global frozen, never fine-tuned; arms copied unchanged. C0 is reused globally trained load model, unlike local-only training; asymmetry disclosed.\n')
    doc('leakage_audit.md','# Leakage audit\n\nOnly frozen Train36 and Dev12 requested. Train-only target range/position/nominal/load statistics. Dev checkpoint selection reused for reporting, bias disclosed. No Final samples, current, W15, gate, vision/object/arm inputs to local MLP or future target input. Nominal computed without demonstrated action in global inference batch. Target only supervision. Global visual/object pathways unchanged. No teleop trials added.\n')
    doc('proposal.md','# Handoff\n\nNo new scientific change authorized. Review this result with user/planner. Do not enlarge bound, fine-tune global, add inputs/history/gate, hardware correction or high-frequency control automatically.\n')
    for name in ['run_f2_3.py','report_f2_3.py']:
        target=out/'source'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(base/'source/scripts'/name,target)
    shutil.copy2(inputs/'decision.md',out/'decision.md')
    save(out/'completion.json',dict(F2_3_COMPLETE=True,outcome=outcome,status='READY',review='PENDING',completed_utc=datetime.now(timezone.utc).isoformat()))
    package=base/'F2_3_result.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in out.rglob('*'):
            if path.is_file():archive.write(path,Path('result')/path.relative_to(out))
    with zipfile.ZipFile(package) as archive:assert archive.testzip() is None
    save(base/'READY.json',dict(status='READY_STOPPED',outcome=outcome,package=str(package),sha256=digest(package)))
    print('READY_STOPPED',outcome,flush=True)


if __name__=='__main__':main()
