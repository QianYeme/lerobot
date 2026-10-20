"""Prepare clean F2.1 r1 arrays and freeze choices; no training or Final sample reads."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/force/f2_1_load_temporal_encoding_ablation/index1'
OUT = BASE / 'result'
FIELDS = ['shoulder_pan.pos','shoulder_lift.pos','elbow_flex.pos','wrist_flex.pos','wrist_roll.pos','gripper.pos']


def save(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def windows(load):
    """Chronological W15, never crossing episode start; repeat earliest sample."""
    load = np.asarray(load)
    return load[np.maximum(np.arange(len(load))[:,None]+np.arange(-14,1)[None,:],0)]


def extract(state, names):
    if names[:6] != FIELDS or 'gripper.load' not in names:
        raise ValueError('Unexpected original schema')
    return np.asarray(state[:,:6],dtype=np.float32).copy(), np.asarray(state[:,names.index('gripper.load')],dtype=np.float32).copy()


def h0_gripper_l1(pred, target):
    return float(np.abs(np.asarray(pred)[:,0,5]-np.asarray(target)[:,5]).mean())


def main():
    if (OUT/'r1_preparation.json').exists():
        raise RuntimeError('Already prepared; do not silently refreeze')
    archive = OUT/'r0_preflight'
    archive.mkdir(exist_ok=False)
    for path in list(OUT.iterdir()):
        if path.is_file():
            shutil.copy2(path,archive/path.name)
    save(OUT/'r0_blocked_record.json',dict(status='BLOCKED_INPUT_SEMANTICS',training_runs=0,archive='r0_preflight/'))
    raw = ROOT/'results/experiments/F2_1_LOAD_TEMPORAL_ENCODING_ABLATION'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    raw.mkdir(parents=True)
    split_path = ROOT/'experiments/data/d1_training_readiness/index1/result/split_manifest_v1.json'
    split = json.loads(split_path.read_text(encoding='utf-8'))
    roles = {part:sorted(r['episode_index'] for r in split['episodes'] if r['split']==part) for part in ['train','dev']}
    assert len(roles['train'])==36 and len(roles['dev'])==12 and not set(roles['train'])&set(roles['dev'])
    dataset = ROOT/'数据集/formal4/kind_merged_nomaster'
    info = json.loads((dataset/'meta/info.json').read_text(encoding='utf-8'))
    names = info['features']['observation.state']['names']
    allowed = roles['train']+roles['dev']
    df = pd.read_parquet(dataset/'data/chunk-000/file-000.parquet',filters=[('episode_index','in',allowed)])
    assert set(df.episode_index)==set(allowed)
    cache, counts = {}, {}
    for ep in allowed:
        frame = df[df.episode_index==ep].sort_values('frame_index')
        assert np.array_equal(frame.frame_index,np.arange(len(frame)))
        state,load = extract(np.stack(frame['observation.state']),names)
        action = np.stack(frame.action).astype(np.float32)
        assert state.shape[1]==action.shape[1]==6
        assert all(np.isfinite(x).all() for x in [state,load,action])
        cache[ep]=(state,load,action)
        counts[str(ep)]=len(frame)
    train_load = np.concatenate([cache[ep][1] for ep in roles['train']]).astype(np.float64)
    mean,std = float(train_load.mean()),float(train_load.std(ddof=0))
    assert std>0
    train_state = np.concatenate([cache[ep][0] for ep in roles['train']]).astype(np.float64)
    train_action = np.concatenate([cache[ep][2] for ep in roles['train']]).astype(np.float64)
    normalization = dict(load=dict(mean=mean,std=std,ddof=0,clipping=None,missing='reject nonfinite; no imputation',fit_episodes=roles['train']),
        state=dict(mean=train_state.mean(0).tolist(),std=train_state.std(0).tolist()),
        action=dict(mean=train_action.mean(0).tolist(),std=train_action.std(0).tolist()),
        images='same ImageNet statistics as historical P1',all_fitted_statistics_scope='Train36 only')
    for ep,(state,load,action) in cache.items():
        normalized=((load-mean)/std).astype(np.float32)
        history=windows(normalized)
        assert np.array_equal(history[:,-1],normalized)
        np.savez_compressed(raw/f'ep{ep:03d}.npz',state=state,action=action,load_single=normalized[:,None],load_w15=history,
                            frames=np.arange(len(load)),episode=ep)
    # Reproduce established metric aggregation, not a new policy forward.
    g0_path = ROOT/'experiments/position/g0_generalization_screening/index1/result/stratified_metrics.json'
    g0 = json.loads(g0_path.read_text(encoding='utf-8'))['P1_OCACT']
    values=[r['metrics']['1']['gripper'] for r in g0['sample_rows']]
    score=float(np.mean(values))
    groups=[v for k,v in g0['overall'].items() if k.startswith('horizontal:')]
    reference=sum(r['gripper_l1']*r['n'] for r in groups)/sum(r['n'] for r in groups)
    assert len(values)==g0['frames']==5748 and np.isclose(score,reference,atol=1e-10)
    save(OUT/'primary_metric_reproduction.json',dict(status='CACHED_METRIC_REDUCTION_PASS',frames=len(values),
        normalized_h0_gripper_l1=score,existing_group_weighted_reference=reference,
        formula='mean_over_all_Dev_frames(abs(pred_normalized[:,0,5]-target_normalized[:,5]))',
        existing_paths=['scripts/eval_g0_generalization.py:main','scripts/complete_g0_report.py:main'],
        ensemble=None,relative_clipping=None,direction='lower is better',
        source_sha256=hashlib.sha256(g0_path.read_bytes()).hexdigest(),
        full_model_forward_reproduction='PENDING_GPU_ENVIRONMENT; cached reduction is not a forward test'))
    save(OUT/'state_semantics.json',dict(revision='r1',fields=FIELDS,shape=[6],load_in_base=False,current_in_any_variant=False,
        A0='state6 + original cameras only',A1='state6 + normalized load[t] as separate input',
        A2='state6 + normalized load[t-14:t] as separate input',startup='left-pad episode-first load; same Train/Dev rule',
        load_normalization=normalization['load'],vae_state='same clean state6; no actuator feedback in VAE state path'))
    save(OUT/'normalization.json',normalization)
    save(OUT/'split_manifest.json',dict(train=roles['train'],dev=roles['dev'],final_count=12,final_sealed=True,
        source=str(split_path.relative_to(ROOT)),sha256=hashlib.sha256(split_path.read_bytes()).hexdigest()))
    configs=OUT/'variant_configs'
    configs.mkdir(exist_ok=True)
    for variant,width in [('A0',0),('A1',1),('A2',15)]:
        representation=None if width==0 else dict(input_width=width,mlp=[width,32,512],activation='GELU',
            parameters=32*width+32+32*512+512,extra_token_position='after robot state token, before image tokens',
            layer_norm=False,dropout=False,contact_gate=False)
        save(configs/f'{variant}.json',dict(variant=variant,revision='r1',state_fields=FIELDS,load_representation=representation,
            chunk_size=100,n_action_steps=1,temporal_ensemble_coeff=None,action_dim=6,gripper_loss_weight=3.,
            optimizer='AdamW',lr=1e-5,lr_backbone=1e-4,weight_decay=1e-4,grad_clip_norm=10.,scheduler=None,
            steps=100000,batch_size=4,seed=1,save_freq=20000,checkpoint_selection='minimum Dev12 normalized h0 gripper L1 at 20k/40k/60k/80k/100k; tie earlier',
            initialization='same ResNet18 ImageNet and frozen DINO/prototype; ACT random seed1; no P_hist warmstart; load layers initialized AFTER common backbone with separate fixed seed1',
            common_shared_tensor_equality='MUST verify exact hashes before first optimizer step',
            current=False,master=False,parameter_status='formula only, actual model instantiation pending',evidence_scope='SINGLE_SEED_EXPLORATORY'))
    preparation=dict(revision='r1',raw=str(raw),episode_frames=counts,train_frames=len(train_load),
        dev_frames=sum(counts[str(ep)] for ep in roles['dev']),source_dataset=str(dataset),
        contract_sha256=hashlib.sha256((BASE/'decision_F2_1_LOAD_TEMPORAL_ENCODING_ABLATION_r1.md').read_bytes()).hexdigest(),
        training_launched=False,model_implemented=False,server='last known westb:27435 connection refused',
        blocked='REMOTE_GPU_UNAVAILABLE',final_content_read=False)
    save(OUT/'r1_preparation.json',preparation)
    save(OUT/'metrics.json',dict(OVERALL_STATUS='PARTIAL_PASS',F2_1_COMPLETE=False,SCIENTIFIC_OUTCOME='UNRESOLVED',
        revision='r1',input_semantics_resolved=True,training_runs=0,blocker='REMOTE_GPU_UNAVAILABLE',
        model_smoke='NOT_RUN',primary_metric_full_forward='NOT_RUN'))
    shutil.copy2(__file__,raw/Path(__file__).name)
    for path in [OUT/'normalization.json',OUT/'state_semantics.json',OUT/'split_manifest.json']:
        shutil.copy2(path,raw/path.name)
    print(json.dumps(dict(train_frames=len(train_load),dev_frames=preparation['dev_frames'],metric=score,raw=str(raw))))


if __name__=='__main__':
    main()
