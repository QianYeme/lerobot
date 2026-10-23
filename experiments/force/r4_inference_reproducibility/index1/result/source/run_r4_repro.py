"""R4 frozen-checkpoint inference probes. Never trains or changes old evidence."""
import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from run_f5_1_r3 import Full60, WORK as F5, OLD, checkpoint, load_policy, digest, loader

WORK = Path('/root/autodl-tmp/R4_20261023')
OUT = WORK / 'result'
LIMIT = .001


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def table(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def tensor_hash(value):
    a = value.detach().cpu().contiguous().numpy() if torch.is_tensor(value) else np.asarray(value)
    return hashlib.sha256(str((a.shape, a.dtype)).encode() + a.tobytes()).hexdigest()


def configure(mode):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = mode in ['historical', 'tf32_off']
    torch.backends.cuda.matmul.allow_tf32 = mode != 'tf32_off'
    torch.backends.cudnn.allow_tf32 = True
    if mode == 'deterministic':
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def environment(policy, seed, mode):
    cp=OLD if seed==36 else checkpoint(seed)
    return dict(mode=mode, checkpoint=str(cp), checkpoint_sha256=digest(cp/'model.safetensors'),
        python_random_state_sha256=hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        numpy_random_state_sha256=hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
        torch_initial_seed=torch.initial_seed(), torch_rng_sha256=tensor_hash(torch.get_rng_state()),
        cuda_initial_seed=torch.cuda.initial_seed(), cuda_rng_sha256=tensor_hash(torch.cuda.get_rng_state()),
        explicit_manual_seed='NONE (historical replay did not set one)', dataloader_worker_seed='torch initial seed + worker id; no random augmentations configured',
        model_training=policy.training, training_dropout=[n for n,m in policy.named_modules() if isinstance(m, torch.nn.Dropout) and m.training],
        amp=torch.is_autocast_enabled(), cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        torch_version=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(), temporal_ensemble_coeff=policy.config.temporal_ensemble_coeff,
        inference='predict_action_chunk; direct stateless chunk, never select_action',
        augmentation='dataset replay + deterministic ImageNet normalization; no train transforms')


def flatten(value, prefix=''):
    if torch.is_tensor(value):
        return {prefix: tensor_hash(value)}
    if isinstance(value, dict):
        return {k2:v2 for k,v in value.items() for k2,v2 in flatten(v, prefix+'/'+str(k)).items()}
    if isinstance(value, (list,tuple)):
        return {k2:v2 for k,v in enumerate(value) for k2,v2 in flatten(v, prefix+'/'+str(k)).items()}
    return {}


def forward(policy, batch, exp):
    features = {}
    handles = []
    for name, module in policy.named_modules():
        if name in ['model.backbone', 'model.object_conditioning', 'model.encoder', 'model.decoder']:
            def hook(_module, _inputs, output, name=name):
                features[name] = flatten(output)
            handles.append(module.register_forward_hook(hook))
    before = flatten(batch)
    queue_before = len(getattr(policy, '_action_queue', []))
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(batch)
        native = chunk * exp.std['action'] + exp.mean['action']
    for handle in handles: handle.remove()
    assert before == flatten(batch), 'MODEL_INPUT_MUTATED'
    assert torch.isfinite(native).all()
    return dict(normalized=chunk.cpu().numpy(), native=native.cpu().numpy()), dict(features=features,
        input_hash=before, queue_before=queue_before, queue_after=len(getattr(policy,'_action_queue',[])),
        temporal_ensembler_present=hasattr(policy,'temporal_ensembler'))


def compare(a, b):
    delta = np.abs(a-b)
    return dict(max_native=float(delta.max()), mean_native=float(delta.mean()), p95_native=float(np.quantile(delta,.95)),
        h0_max_native=float(delta[:,0].max()), per_joint_max=delta.max(axis=tuple(range(delta.ndim-1))).tolist(),
        passed=bool(np.isfinite(delta).all() and delta.max()<LIMIT))


def fixture_paths():
    return sorted((WORK/'fixtures').glob('*.pt'))


def prepare(exp):
    with np.load(F5/'cache/G60-S3.npz') as a, np.load(F5/'cache/G60-S3_repeat.npz') as b:
        assert np.array_equal(a['episode'], b['episode']) and np.array_equal(a['frame'], b['frame'])
        delta = np.abs(a['prediction_native']-b['prediction_native'])
        worst = int(np.unravel_index(np.argmax(delta), delta.shape)[0])
        labels = ['first', 'historical_worst', 'tail']
        groups = [list(range(4)), list(range(worst//4*4, min(worst//4*4+4,len(delta)))), list(range(len(delta)//4*4,len(delta)))]
        historical = dict(max_native=float(delta.max()), mean_native=float(delta.mean()), p95_native=float(np.quantile(delta,.95)),
            worst_row=worst, worst_episode=int(a['episode'][worst]), worst_frame=int(a['frame'][worst]),
            per_joint_max=delta.max(0).tolist(), cache_normal_sha256=digest(F5/'cache/G60-S3.npz'),
            cache_repeat_sha256=digest(F5/'cache/G60-S3_repeat.npz'))
        ids = list(zip(a['episode'].tolist(), a['frame'].tolist()))
    save(OUT/'historical_divergence.json', historical)
    dataset = exp.loader('replay',workers=0).dataset
    manifest=[]
    for name, indices in zip(labels,groups):
        raw=next(iter(DataLoader(Subset(dataset,indices),batch_size=len(indices),shuffle=False,num_workers=0)))
        actual=list(zip(raw['episode_index'].reshape(-1).tolist(),raw['frame_index'].reshape(-1).tolist()))
        assert actual == [ids[i] for i in indices]
        batch={k:v.detach().cpu() for k,v in exp.batch(raw,'A1').items() if k!='action'}
        path=WORK/'fixtures'/f'{name}.pt';path.parent.mkdir(exist_ok=True)
        torch.save(dict(batch=batch,indices=indices,ids=actual),path)
        manifest.append(dict(name=name,indices=indices,ids=actual,sha256=digest(path),tensors=flatten(batch)))
    save(OUT/'frozen_input_manifest.json',manifest)


def probe(args):
    exp=Full60(args.seed)
    if not fixture_paths(): prepare(exp)
    policy=load_policy(exp,checkpoint(args.seed));policy.reset()
    prefix=WORK/'raw'/args.mode/f'S{args.seed}'
    prefix.mkdir(parents=True,exist_ok=True)
    save(prefix/'audit_config.json',environment(policy,args.seed,args.mode))
    records=[]
    for path in fixture_paths():
        data=torch.load(path,map_location='cpu',weights_only=True)
        batch={k:v.cuda() for k,v in data['batch'].items()}
        outputs=[]
        for repeat in range(3):
            policy.reset()
            values,meta=forward(policy,batch,exp)
            np.savez_compressed(prefix/f'{path.stem}_A{repeat}.npz',**values)
            save(prefix/f'{path.stem}_A{repeat}.json',meta)
            outputs.append(values['native'])
        for repeat in [1,2]:
            records.append(dict(seed=args.seed,mode=args.mode,fixture=path.stem,comparison=f'A0_vs_A{repeat}',**compare(outputs[0],outputs[repeat])))
    table(prefix/'same_process_repeat.csv',records)
    print(json.dumps(records),flush=True)


def infer(args):
    exp=Full60(args.seed);policy=load_policy(exp,checkpoint(args.seed));policy.reset()
    prefix=WORK/'raw'/args.mode/f'S{args.seed}'/args.tag
    prefix.mkdir(parents=True,exist_ok=False)
    save(prefix/'audit_config.json',environment(policy,args.seed,args.mode))
    for path in fixture_paths():
        data=torch.load(path,map_location='cpu',weights_only=True)
        batch={k:v.cuda() for k,v in data['batch'].items()}
        values,meta=forward(policy,batch,exp)
        np.savez_compressed(prefix/f'{path.stem}.npz',**values)
        save(prefix/f'{path.stem}.json',meta)
    print('B_INFERENCE_DONE',args.seed,args.mode,args.tag,flush=True)


def cross(args):
    prefix=WORK/'raw'/args.mode/f'S{args.seed}'
    with (prefix/'same_process_repeat.csv').open() as stream:
        assert all(r['passed']=='True' for r in csv.DictReader(stream)), 'A_FAILED_STOP_B'
    for tag in ['B0','B1']:
        with (WORK/'logs'/f'S{args.seed}_{args.mode}_{tag}.log').open('w') as stream:
            subprocess.run([sys.executable,__file__,'--stage','infer','--seed',str(args.seed),
                '--mode',args.mode,'--tag',tag],stdout=stream,stderr=subprocess.STDOUT,check=True)
    records=[]
    for path in fixture_paths():
        a=np.load(prefix/'B0'/f'{path.stem}.npz')['native']
        b=np.load(prefix/'B1'/f'{path.stem}.npz')['native']
        meta0=json.loads((prefix/'B0'/f'{path.stem}.json').read_text())
        meta1=json.loads((prefix/'B1'/f'{path.stem}.json').read_text())
        assert meta0['input_hash']==meta1['input_hash']
        different=[k for k in meta0['features'] if meta0['features'][k]!=meta1['features'].get(k)]
        records.append(dict(seed=args.seed,mode=args.mode,fixture=path.stem,
            first_different_hook=different[0] if different else 'NONE',input_bitwise_equal=True,**compare(a,b)))
    table(prefix/'cross_process_repeat.csv',records)
    print(json.dumps(records),flush=True)


def preprocess(args):
    prefix=WORK/'raw'/args.mode/f'S{args.seed}'
    with (prefix/'cross_process_repeat.csv').open() as stream:
        assert all(r['passed']=='True' for r in csv.DictReader(stream)), 'B_FAILED_STOP_C'
    policy=load_policy(Full60(args.seed),checkpoint(args.seed))
    records=[]; observed={}
    for repeat in [0,1]:
        exp=Full60(args.seed);dataset=exp.loader('replay',workers=0).dataset
        for path in fixture_paths():
            fixture=torch.load(path,map_location='cpu',weights_only=True)
            raw=next(iter(DataLoader(Subset(dataset,fixture['indices']),batch_size=len(fixture['indices']),
                shuffle=False,num_workers=4,pin_memory=True)))
            ids=list(zip(raw['episode_index'].reshape(-1).tolist(),raw['frame_index'].reshape(-1).tolist()))
            assert ids==fixture['ids']
            batch={k:v for k,v in exp.batch(raw,'A1').items() if k!='action'}
            stages={'sample_ids':hashlib.sha256(repr(ids).encode()).hexdigest(),
                'decoded_images':{k:tensor_hash(raw[k]) for k in ['observation.images.top','observation.images.gripper']},
                'raw_dataset_state':tensor_hash(raw['observation.state']),
                'source_state_array':tensor_hash(np.stack([exp.cache[e]['state'][f] for e,f in ids])),
                'normalized_state':tensor_hash(batch['observation.state']),
                'normalized_load':tensor_hash(batch['observation.load']),
                'normalized_images':{k:tensor_hash(batch[k]) for k in ['observation.images.top','observation.images.gripper']},
                'model_inputs':flatten(batch)}
            policy.reset();values,meta=forward(policy,batch,exp)
            stages['feature_hooks']=meta['features']
            save(prefix/f'{path.stem}_C{repeat}.json',stages)
            np.savez_compressed(prefix/f'{path.stem}_C{repeat}.npz',**values)
            if repeat==0: observed[path.stem]=(stages,values['native'])
            else:
                previous,output=observed[path.stem]
                first=next((k for k in stages if stages[k]!=previous[k]),'NONE')
                records.append(dict(seed=args.seed,mode=args.mode,fixture=path.stem,first_divergent_stage=first,
                    inputs_bitwise_equal=stages['model_inputs']==previous['model_inputs'],**compare(output,values['native'])))
    table(prefix/'preprocess_repeat.csv',records)
    assert all(r['inputs_bitwise_equal'] and r['passed'] for r in records),'C_FAILED_STOP_D'
    print('C_PASS',json.dumps(records),flush=True)


def full(args):
    prefix=WORK/'raw'/args.mode/f'S{args.seed}'
    if args.tag=='POSTFIX':
        assert json.loads((OUT/'repro_gate.json').read_text())['R4_REPRODUCIBILITY_PASS'], 'CORE_REPRO_NOT_PASSED'
    else:
        with (prefix/'preprocess_repeat.csv').open() as stream:
            assert all(r['passed']=='True' and r['inputs_bitwise_equal']=='True' for r in csv.DictReader(stream)), 'C_NOT_PASSED'
    destination=prefix/args.tag;destination.mkdir(parents=True,exist_ok=False)
    exp=Full60(3 if args.seed==36 else args.seed,old=args.seed==36)
    cp=OLD if args.seed==36 else checkpoint(args.seed)
    policy=load_policy(exp,cp);policy.reset()
    save(destination/'audit_config.json',environment(policy,args.seed,args.mode))
    records={k:[] for k in ['episode','frame','state','raw_load','target_native','prediction_native']}
    if args.seed==3: records['neutral_native']=[]
    hashes=[];chunks=destination/'chunks';chunks.mkdir()
    with torch.inference_mode():
        for index,raw in enumerate(loader(exp)):
            batch=exp.batch(raw,'A1');batch.pop('action');policy.reset()
            chunk=policy.predict_action_chunk(batch)
            native=chunk*exp.std['action']+exp.mean['action']
            assert torch.isfinite(native).all()
            episodes=raw['episode_index'].reshape(-1).tolist();frames=raw['frame_index'].reshape(-1).tolist()
            ids=list(zip(episodes,frames))
            np.savez_compressed(chunks/f'batch_{index:05d}.npz',native=native.cpu().numpy())
            # Hash all model inputs, not just IDs, to expose data drift between passes.
            hashes.append(dict(batch=index,ids=ids,inputs=flatten(batch),queue_length=len(getattr(policy,'_action_queue',[]))))
            values=dict(episode=episodes,frame=frames,state=[exp.cache[e]['state'][f] for e,f in ids],
                raw_load=[exp.cache[e]['load_single'][f,0]*exp.norm['load']['std']+exp.norm['load']['mean'] for e,f in ids],
                target_native=raw['action'][:,0].numpy(),prediction_native=native[:,0].cpu().numpy())
            if args.seed==3:
                cf=dict(batch);cf['observation.load']=torch.zeros_like(batch['observation.load'])
                neutral=policy.predict_action_chunk(cf)[:,0]*exp.std['action']+exp.mean['action']
                values['neutral_native']=neutral.cpu().numpy()
            for k,v in values.items():records[k].append(np.asarray(v))
            if index%500==0: print('D_REPLAY',args.seed,args.mode,args.tag,index,flush=True)
    result={k:np.concatenate(v) for k,v in records.items()}
    assert len(result['episode'])==28739
    assert len(set(zip(result['episode'],result['frame'])))==28739
    np.savez_compressed(destination/'replay.npz',**result)
    save(destination/'input_manifest.json',hashes)
    save(destination/'complete.json',dict(status='DONE',frames=28739,batches=len(hashes),checkpoint_sha256=digest(cp/'model.safetensors')))
    print('D_DONE',args.seed,args.mode,args.tag,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['probe','infer','cross','preprocess','full'],required=True)
    ap.add_argument('--mode',choices=['historical','benchmark_off','tf32_off','deterministic'],default='historical')
    ap.add_argument('--seed',type=int,choices=[1,2,3,4,36],default=3);ap.add_argument('--tag',default='B0')
    args=ap.parse_args();configure(args.mode)
    {'probe':probe,'infer':infer,'cross':cross,'preprocess':preprocess,'full':full}[args.stage](args)


if __name__=='__main__': main()
