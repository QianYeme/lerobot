"""P4.2 frozen sequential single-GPU queue; reuses the exact P4.1 input path."""
import hashlib
import copy
import json
import random
import shutil
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

import p4_1_pilot as old

ROOT = old.ROOT
P41 = old.OUT
WORK = ROOT / 'experiments/p4_final_p1_training/p4_2_unattended_multi_model/index1'
OUT = WORK / 'result'
BASE = P41 / 'checkpoints/pilot/080000/resume.pt'


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def append(name, value):
    previous = old.OUT
    old.OUT = OUT
    try:
        old.row(name, value)
    finally:
        old.OUT = previous


def reseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batches(dataset, seed, epoch, cursor):
    indices = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(dataset)).tolist()
    indices = indices[:len(indices)//4*4][cursor*4:]
    return DataLoader(Subset(dataset, indices), batch_size=4, num_workers=4, shuffle=False,
                      generator=torch.Generator().manual_seed(seed + epoch), pin_memory=True)


def parameter_hash(policy):
    digest = hashlib.sha256()
    for name, tensor in policy.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def update(policy, optimizer, batch):
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    loss, _ = policy(batch)
    assert torch.isfinite(loss), 'NONFINITE_LOSS'
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10)
    assert torch.isfinite(grad), 'NONFINITE_GRAD'
    optimizer.step()
    return float(loss.detach()), float(grad)


def evaluate(policy, pre, stats, ids, name, step, p1):
    dataset = old.Mapped(ids)
    meta = pd.read_parquet(old.DATA / 'meta/episodes/chunk-000/file-000.parquet')
    offset = 0
    gate = policy.model.object_conditioning.gate if p1 else None
    trace = {}
    def hook(_module, args, output):
        trace['residual'] = float((gate.detach()*output[0].detach()).norm())
    handle = policy.model.object_conditioning.cross_attention.register_forward_hook(hook) if p1 else None
    try:
        for ep in ids:
            length = int(meta[meta.episode_index == ep].iloc[0].length)
            for fraction in (0, .15, .3, .5, .7, .9):
                raw = dataset[offset + min(length-1, int(fraction*length))]
                batch = pre(raw)
                with torch.inference_mode():
                    pred = policy.predict_action_chunk(batch)
                    residual = trace.get('residual')
                    if p1:
                        saved_gate = gate.detach().clone()
                        gate.zero_()
                        try:
                            off = policy.predict_action_chunk(batch)
                        finally:
                            gate.copy_(saved_gate)
                valid = ~batch['action_is_pad'][0]
                error = (pred[0]-batch['action'][0]).abs()[valid]
                assert torch.isfinite(error).all()
                physical = error*stats['action']['std'].cuda()
                h0 = pred[0,0]*stats['action']['std'].cuda()+stats['action']['mean'].cuda()
                append('dev15_model_comparison.csv', dict(run=name, step=step, episode=ep, progress_proxy=fraction,
                    position_bucket='NOT_VERIFIED_NO_XY_GT', arm_normalized=float(error[:,:5].mean()),
                    gripper_normalized=float(error[:,5].mean()), arm_raw_mae=float(physical[:,:5].mean()),
                    gripper_raw_mae=float(physical[:,5].mean()), h0_pan_delta=float(h0[0]-raw['observation.state'][0].cuda())))
                if p1:
                    difference=float((pred-off).abs().mean())
                    append('p1_object_path_diagnostics.csv',dict(run=name,step=step,episode=ep,progress_proxy=fraction,
                        on_off_mean_abs=difference,gate_alpha=float(saved_gate),residual_norm=residual))
            offset += length
    finally:
        if handle:
            handle.remove()


def save(policy, optimizer, cfg, run, step, epoch, cursor):
    path=old.checkpoint(policy,optimizer,step,epoch,cursor,cfg,'formal')
    append('checkpoint_manifest.csv',dict(run=run,step=step,path=str(path),sha256=old.sha(path),complete_resume=True))
    return path


def run(name, seed, p1, continued, train, dev, cutoff):
    folder=OUT/'runs'/name
    folder.mkdir(parents=True,exist_ok=False)
    old.OUT=folder
    shutil.copyfile(P41/'formal6_train90_prototypes.pt',folder/'formal6_train90_prototypes.pt')
    shutil.copyfile(P41/'prototype_provenance.json',folder/'prototype_provenance.json')
    stats=torch.load(P41/'train90_statistics.pt',map_location='cpu',weights_only=False)
    torch.save(stats,folder/'train90_statistics.pt')
    reseed(seed)
    cfg=old.config()
    if not p1:
        cfg.model_type='act'
        cfg.object_conditioning_enabled=False
        cfg.object_conditioning_prototype_path=None
    cfg.save_pretrained(folder/'effective_model_config')
    dataset=old.Mapped(train)
    pre,post=old.make_act_pre_post_processors(cfg,stats)
    policy=old.ACTPolicy(cfg).cuda()
    optimizer=torch.optim.AdamW(policy.get_optim_params(),lr=1e-5,weight_decay=1e-4)
    step,epoch,cursor=0,0,0
    target=160000 if continued else 80000
    if continued:
        state=torch.load(BASE,map_location='cpu',weights_only=False)
        assert state['step']==80000 and state['scheduler'] is None
        assert state['dataset_revision']==old.REV
        assert state['split_sha']==old.sha(old.PRIOR/'split_freeze_manifest.json')
        assert state['prototypes_sha']==old.sha(P41/'formal6_train90_prototypes.pt')
        policy.load_state_dict(state['model'])
        optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        step,epoch,cursor=state['step'],state['epoch'],state['next_batch_cursor']
        # Repeated disposable next-step branches validate exact restored model/optimizer/RNG path.
        raw=next(iter(batches(dataset,seed,epoch,cursor)))
        batch=pre(raw)
        branches=[]
        for _ in range(2):
            policy.load_state_dict(state['model'])
            optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
            old.restore(state['rng'])
            loss,grad=update(policy,optimizer,batch)
            branches.append(dict(loss=loss,grad=grad,hash=parameter_hash(policy)))
        assert branches[0]==branches[1], 'RESUME_NOT_EXACT'
        write('resume_integrity.json',dict(status='EXACT_BRANCH_REPLAY_PASS',branches=branches,source_sha256=old.sha(BASE),
             sampler='same SeedSequence([seed,epoch]), saved consumed batch cursor, no augmentation',probe_steps_discarded=2))
        policy.load_state_dict(state['model'])
        optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        old.restore(state['rng'])
        del state
    append('run_manifest.csv',dict(run=name,seed=seed,p1=p1,input_state_dim=8,action_dim=6,initial_step=step,
        target_step=target,continuation=continued,stats_sha256=old.sha(P41/'train90_statistics.pt'),
        prototype_sha256=old.sha(P41/'formal6_train90_prototypes.pt') if p1 else 'NOT_USED',
        lr=1e-5,batch=4,clip=10,weight_decay=.0001,chunk=100,source_commit='4e9e80a342fc4808811daba7d4e4ad9cca799e73'))
    points=(100000,120000,140000,160000) if continued else (20000,40000,60000,80000)
    started=time.monotonic()
    initial=step
    last_save=time.monotonic()
    checkpoints=[]
    while step<target and time.monotonic()<cutoff:
        for raw in batches(dataset,seed,epoch,cursor):
            batch=pre(raw)
            loss,grad=update(policy,optimizer,batch)
            torch.cuda.synchronize()
            step+=1
            cursor+=1
            old.row('training_metrics.csv',dict(step=step,epoch=epoch,cursor=cursor,loss=loss,grad_norm=grad,
                elapsed_s=time.monotonic()-started,peak_memory_bytes=torch.cuda.max_memory_allocated()))
            if step-initial==500:
                append('throughput_and_resource.csv',dict(run=name,steps=500,elapsed_s=time.monotonic()-started,
                    sec_per_step=(time.monotonic()-started)/500,peak_memory_bytes=torch.cuda.max_memory_allocated()))
            if step in points or time.monotonic()-last_save>=1800 or time.monotonic()>=cutoff:
                checkpoints.append((step,save(policy,optimizer,cfg,name,step,epoch,cursor)))
                last_save=time.monotonic()
            if step>=target or time.monotonic()>=cutoff:
                break
        else:
            epoch+=1
            cursor=0
            continue
        break
    if not checkpoints or checkpoints[-1][0]!=step:
        checkpoints.append((step,save(policy,optimizer,cfg,name,step,epoch,cursor)))
    # Always evaluate the last checkpoint. P0 also evaluates 80k and 120k when available.
    selected=[checkpoints[-1]]
    if continued:
        selected=[(80000,BASE)]+[(s,p) for s,p in checkpoints if s==120000]+selected
    for saved_step,path in selected:
        value=torch.load(path,map_location='cpu',weights_only=False)
        policy.load_state_dict(value['model'])
        del value
        evaluate(policy,pre,stats,dev,name,saved_step,p1)
    append('queue_execution.csv',dict(run=name,status='COMPLETE' if step==target else 'PARTIAL_BUDGET',
        initial_step=initial,final_step=step,elapsed_s=time.monotonic()-started))
    del policy,optimizer,pre,post,dataset
    torch.cuda.empty_cache()
    return step


def package():
    (OUT/'summary.md').write_text('# P4.2 result\n\nSee queue_execution.csv for actual completion/partial/failure; no real robot or Final inference. Independent reviews PENDING. No weights are included in this zip.\n')
    (OUT/'validation.md').write_text('# Validation\n\nP4.1 production input/normalization gates reused. Exact resume replay evidence in resume_integrity.json when successful. All models use state8D/action6D, Train-only statistics and the same frozen manifests. Model configs and run logs retained on server.\n')
    (OUT/'resume_integrity.md').write_text('# Resume integrity\n\nSee resume_integrity.json. No fresh run substitutes P0. Two discarded next-step probes replay restored optimizer/RNG/sampler, then original80k state is restored before formal continuation.\n')
    (OUT/'newgl_comparability.md').write_text('# New-GL\n\nSKIPPED_INCOMPATIBLE. Historical run_f4_2.py uses separate strong-global6D+load and frozen-global local3-32-32-1 residual trained10000 steps, Adam .001, plus calibrated residual bound. This is not a single end-to-end80k8D model. New formal6 global/local budget and bound adaptation are not unambiguously frozen; do not invent a substitute. Conditional skip is authorized; request a separate reconstruction contract.\n')
    (OUT/'final15_access_audit.md').write_text('# Final15\n\nThis P4.2 queue never constructs a Final dataset or reads Final images/scalar rows for training, evaluation, prototypes or selection. Manifest hashes only. Historical P4.1 raw-scalar access exception remains disclosed in its report; no new access here.\n')
    for name in ('builder_review.md','red_team_review.md'):
        (OUT/name).write_text('# Independent review\n\nPENDING. Inspect exact resume, seed initialization, same8D ACT baseline, Train-only statistics/prototypes, partial budgets and Dev sample limits.\n')
    (OUT/'proposal.md').write_text('# Next decision\n\nReview sampled Dev comparisons before selecting minimal robot candidates; no automatic robot, additional seed, changed budget or New-GL approximation. New-GL reconstruction requires explicit two-stage budget/interface/bound contract.\n')
    if (OUT/'dev15_model_comparison.csv').exists():
        frame=pd.read_csv(OUT/'dev15_model_comparison.csv')
        frame.groupby(['run','step','episode'])[['arm_normalized','gripper_normalized','arm_raw_mae','gripper_raw_mae']].agg(['mean','median']).to_csv(OUT/'dev15_per_episode_summary.csv')
    if (OUT/'p1_object_path_diagnostics.csv').exists():
        frame=pd.read_csv(OUT/'p1_object_path_diagnostics.csv')
        frame.groupby(['run','step','episode']).on_off_mean_abs.quantile([0,.25,.5,.75,1]).to_csv(OUT/'p1_object_path_quantiles.csv')
        write('object_near_zero_definition.json',dict(threshold=1e-6,interpretation='numerical-zero diagnostic only, not semantic failure threshold',
            ratio=float((frame.on_off_mean_abs<=1e-6).mean())))
    delivery=WORK/'delivery_small'
    delivery.mkdir(exist_ok=True)
    for path in OUT.iterdir():
        if path.is_file():
            shutil.copyfile(path,delivery/path.name)
    shutil.copyfile(Path(__file__),delivery/'p4_2_queue.py')
    hashes={p.name:old.sha(p) for p in delivery.iterdir() if p.is_file() and p.name!='delivery_manifest.json'}
    (delivery/'delivery_manifest.json').write_text(json.dumps(hashes,indent=2))
    with zipfile.ZipFile(WORK/'P4_2_r0_result_delivery.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in delivery.iterdir():
            archive.write(path,'result/'+path.name)
    with zipfile.ZipFile(WORK/'P4_2_r0_result_delivery.zip') as archive:
        assert archive.testzip() is None


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'queue_started.json').exists(), 'Refuse duplicate queue'
    start=time.monotonic()
    cutoff=start+8*3600
    write('queue_started.json',dict(start_unix=time.time(),training_budget_s=8*3600,delivery_deadline_s=9*3600,
        hard_limit_s=10*3600,lease='user confirms no expiry',scheduling='single GPU sequential',core_priority=['P0','P1','P2','P3']))
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(False)
    train=json.loads((old.PRIOR/'train90_episode_ids.json').read_text())
    dev=json.loads((old.PRIOR/'dev15_episode_ids.json').read_text())
    write('data_and_source_identity.json',dict(dataset_revision=old.REV,
        split_hashes={name:old.sha(old.PRIOR/name) for name in ['train90_episode_ids.json','dev15_episode_ids.json','final15_episode_ids.json']},
        prototype_sha=old.sha(P41/'formal6_train90_prototypes.pt'),stats_sha=old.sha(P41/'train90_statistics.pt'),
        source_sha=old.sha(__file__),helper_sha=old.sha(ROOT/'scripts/p4_1_pilot.py')))
    for name,seed,p1,continued in [('P0_P1_S1_160k',1,True,True),('P1_ACT_S1_80k',1,False,False),
                                    ('P2_P1_S2_80k',2,True,False),('P3_P1_S3_80k',3,True,False)]:
        if time.monotonic()>=cutoff:
            append('queue_execution.csv',dict(run=name,status='NOT_RUN_TIME',initial_step=0,final_step=0,elapsed_s=0))
            continue
        try:
            run(name,seed,p1,continued,train,dev,cutoff)
        except Exception:
            (OUT/(name+'_failure.log')).write_text(traceback.format_exc())
            append('queue_execution.csv',dict(run=name,status='FAILED_NO_CONFIG_CHANGE',initial_step=80000 if continued else 0,final_step=-1,elapsed_s=time.monotonic()-start))
            torch.cuda.empty_cache()
    package()
    print('P4_2_QUEUE_DELIVERY_READY',flush=True)


if __name__=='__main__':
    main()
