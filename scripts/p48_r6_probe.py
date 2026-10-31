"""Read-only object-path probes. No optimizer is advanced and no model weights are saved."""
import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path('/root/autodl-tmp/lerobot-main')
R5=ROOT/'experiments/p4_public_benchmark/p4_8_libero_pilot/index5'
OUT=R5.parent/'index6/result'
sys.path.insert(0,str(R5/'workspace/scripts'))
from p48_r5_common import CachedData,make_policy,model_hash,seed,sha,write


def csv_out(name,rows):
    with (OUT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


class Trace:
    def __init__(self,policy):
        self.oc=policy.model.object_conditioning;self.data={};self.alpha=None;self.retain=False
        self.handles=[self.oc.cross_attention.register_forward_hook(self.attention),
            self.oc.tokenizer.register_forward_hook(self.tokenizer),self.oc.register_forward_hook(self.fusion),
            policy.model.backbone.register_forward_hook(self.cnn)]
        self.original=self.oc.dino.model.forward_features
        self.oc.dino.model.forward_features=self.dino

    def dino(self,x):
        self.data['dino_input']=x
        result=self.original(x);self.data['dino_features']=result['x_norm_patchtokens'];return result

    def cnn(self,_module,args,output):
        self.data.setdefault('cnn_features',[]).append(output['feature_map'])

    def attention(self,_module,args,output):
        self.data['query']=args[0];self.data['object_tokens']=args[1];self.data['residual']=output[0]
        if self.retain and args[0].requires_grad:args[0].retain_grad()

    def tokenizer(self,_module,_args,output):
        self.data['raw_tokens']=output[0];self.data['similarity']=output[1]

    def fusion(self,_module,args,output):
        if self.alpha is not None:output=args[0]+self.alpha*self.data['residual']
        self.data['encoder_input']=output;return output

    def clear(self):self.data={}

    def close(self):
        for h in self.handles:h.remove()
        self.oc.dino.model.forward_features=self.original


def predict(policy,trace,batch,alpha=None,bypass=False):
    import torch
    trace.clear();trace.alpha=alpha
    original=trace.oc.forward
    if bypass:trace.oc.forward=lambda query,top,object_tokens=None:query
    try:
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            value=policy.predict_action_chunk(batch)
            assert torch.isfinite(value).all(),'Non-finite diagnostic action; do not execute actions'
            return value.detach().cpu().numpy()[0]
    finally:
        trace.alpha=None
        if bypass:trace.oc.forward=original


def differences(value,reference,std):
    import numpy as np
    error=np.abs(value-reference);raw=error*std
    return dict(h0_arm_normalized=float(error[0,:6].mean()),h0_gripper_normalized=float(error[0,6]),
        h0_arm_raw=float(raw[0,:6].mean()),h0_gripper_raw=float(raw[0,6]),
        chunk_arm_normalized_mean=float(error[:,:6].mean()),chunk_gripper_normalized_mean=float(error[:,6].mean()),
        chunk_arm_raw_mean=float(raw[:,:6].mean()),chunk_gripper_raw_mean=float(raw[:,6].mean()),
        chunk_normalized_max=float(error.max()),chunk_raw_max=float(raw.max()),
        discrete_raw_mae=json.dumps({i:raw[i].tolist() for i in [0,1,5,10,25,50,99]}))


def gradient_stats(parameters):
    import torch
    parameters=list(parameters);total=sum(p.numel() for p in parameters)
    present=[p.grad.detach() for p in parameters if p.grad is not None]
    if not present:return dict(l2=0.,zero_fraction=1.,finite_fraction=1.,grad_present_parameters=0,total_elements=total)
    squares=torch.stack([g.double().square().sum() for g in present]).sum()
    nonzero=sum(int(torch.count_nonzero(g)) for g in present)
    finite=sum(int(torch.isfinite(g).sum()) for g in present)
    missing=total-sum(g.numel() for g in present)
    return dict(l2=float(squares.sqrt()),zero_fraction=1-nonzero/max(total,1),
                finite_fraction=(finite+missing)/max(total,1),grad_present_parameters=len(present),total_elements=total)


def gradients(policy,trace,batch,kind,step,split,alpha_override=None):
    import torch
    seed(1);policy.train();policy.model.backbone.eval()
    for module in policy.modules():
        if isinstance(module,(torch.nn.Dropout,torch.nn.MultiheadAttention)):module.eval()
    trace.clear();trace.retain=True
    alpha=None
    if alpha_override is not None:
        alpha=torch.tensor(alpha_override,device='cuda',requires_grad=True);trace.alpha=alpha
    policy.zero_grad(set_to_none=True)
    try:
        loss,parts=policy(batch);assert torch.isfinite(loss);loss.backward()
        gate=trace.oc.gate if alpha is None else alpha
        rows=[]
        groups={'alpha':[gate],'adapter':trace.oc.adapter.parameters(),
                'cross_attention':trace.oc.cross_attention.parameters(),'slots':trace.oc.token_slot_embedding.parameters(),
                'cnn_trunk':policy.model.backbone.parameters(),'encoder_query':[trace.data['query']]}
        for name,parameters in groups.items():
            row=dict(model=kind,checkpoint_step=step,probe_id=split+'_locked_batch8',split=split,
                alpha=float(trace.oc.gate) if alpha is None else alpha_override,group=name,
                loss=float(loss.detach()),l1=parts['l1_loss'],kl=parts.get('kld_loss'),
                units='d(native training loss)/d(parameter); before clipping; no update',
                measurement='CVAE posterior loss, dropoutoff, seed1',eval_dtype='float32',**gradient_stats(parameters))
            assert row['finite_fraction']==1
            if name=='alpha':row['signed_gradient']=float(gate.grad) if gate.grad is not None else None
            else:row['signed_gradient']=None
            rows.append(row)
        return rows
    finally:
        trace.alpha=None;trace.retain=False;policy.zero_grad(set_to_none=True);trace.clear();policy.eval()


def heatmap_record(sample,kind,step,trace):
    import numpy as np
    sim=trace.data['similarity'][0].detach().cpu().numpy()
    tokens=trace.data['raw_tokens'][0].detach().cpu().numpy()
    regions=[sim,sim[:8,:8],sim[:8,8:],sim[8:,:8],sim[8:,8:]]
    weights=[];entropy=[]
    for region in regions:
        exp=np.exp((region-region.max())/.20);w=exp/exp.sum();weights.append(w.flatten().tolist());entropy.append(float(-(w*np.log(w+1e-12)).sum()))
    top=np.argsort(sim.flatten())[-5:][::-1].tolist()
    result=dict(model=kind,checkpoint_step=step,probe_id=sample['id'],episode=sample['episode'],frame=sample['frame'],
        split=sample['split'],role=sample['role'],units='cosine / normalized coordinates / pooling probability',
        measurement='actual online DINO and original tokenizer',eval_dtype='float32',
        similarity_min=float(sim.min()),similarity_max=float(sim.max()),top5_patch_indices=json.dumps(top),
        token_xy_confidence=json.dumps(tokens[:,384:387].tolist()),region_entropy=json.dumps(entropy),
        similarity_sha256=hashlib.sha256(sim.tobytes()).hexdigest(),reference_roi_weight_mass=None)
    if sample['role']=='bank_reference':result['reference_roi_weight_mass']=sum(weights[0][i] for i in [7*16+11,7*16+12])
    payload=dict(**result,similarity=sim.tolist(),region_weights=weights)
    return result,payload


def run(kind,pilot=False):
    import numpy as np
    import torch
    lock=json.loads((OUT/'input_lock.json').read_text());datasets={s:CachedData(s) for s in ['train','dev']}
    samples=lock['samples'];index={s['id']:s for s in samples}
    checkpoint_rows=json.loads((OUT/'upstream_manifest.json').read_text())['checkpoints']
    scales=[];doses=[];audits=[];heat_rows=[];heat_data=[];grad_rows=[];dependencies=[];comparison=[];integrity=[]
    selected_steps=[10000] if pilot else [5000,10000]
    for step in selected_steps:
        cfg,policy=make_policy(kind)
        source=next(r for r in checkpoint_rows if r['kind']==kind and int(r['step'])==step)
        saved=torch.load(source['path'],map_location='cpu',weights_only=False);policy.load_state_dict(saved['model'],strict=True);policy.eval()
        before=model_hash(policy);trace=Trace(policy)
        std=datasets['train'].stats['action']['std'].numpy()+1e-8
        if pilot:
            batch=datasets['dev'].batch([samples[0]['index']]);start=time.perf_counter()
            for _ in range(20):predict(policy,trace,batch)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            write(OUT/f'pilot_{kind}.json',dict(model=kind,forwards=20,wall_seconds=elapsed,forwards_per_second=20/elapsed,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
            trace.close();return
        for sample in samples:
            batch=datasets[sample['split']].batch([sample['index']]);on=predict(policy,trace,batch)
            captured={key:value for key,value in trace.data.items()}
            alpha=float(trace.oc.gate.detach());q=captured['query'];res=captured['residual']
            norms={name:float(value.norm()) for name,value in captured.items() if isinstance(value,torch.Tensor)}
            checks={k:v for k,v in captured.items() if not isinstance(v,list)}
            checks.update({f'cnn_feature_{i}':v for i,v in enumerate(captured['cnn_features'])})
            checks.update({k:batch[k] for k in ['observation.state','observation.images.image','observation.images.image2','_oc_raw_top']})
            for name,value in checks.items():
                for tensor in value if isinstance(value,list) else [value]:
                    assert torch.isfinite(tensor).all(),(sample['id'],name)
            assert np.isfinite(on).all()
            scales.append(dict(model=kind,checkpoint_step=step,probe_id=sample['id'],episode=sample['episode'],frame=sample['frame'],
                split=sample['split'],alpha=alpha,abs_alpha=abs(alpha),query_l2=norms['query'],object_tokens_l2=norms['object_tokens'],
                raw_residual_l2=norms['residual'],gated_residual_l2=float((alpha*res).norm()),encoder_input_l2=norms['encoder_input'],
                gated_to_query=float((alpha*res).norm()/q.norm().clamp_min(1e-12)),
                units='L2 activation norm',measurement='original OC immediately before ACT encoder',eval_dtype='float32',
                shape_dtype_finite=json.dumps({k:[list(v.shape),str(v.dtype),True] for k,v in checks.items()})))
            hrow,hdata=heatmap_record(sample,kind,step,trace);heat_rows.append(hrow);heat_data.append(hdata)
            off=predict(policy,trace,batch,alpha=0.)
            bypass=predict(policy,trace,batch,bypass=True)
            repeated=predict(policy,trace,batch)
            bypass_diff=float(np.abs(off-bypass).max());repeat_diff=float(np.abs(on-repeated).max())
            assert bypass_diff<=1e-7 and repeat_diff==0
            measure=differences(on,off,std);old_error=None
            if step==10000 and sample['split']=='dev':
                with (R5/'result/object_path_diagnostics.csv').open() as stream:
                    old=next(r for r in csv.DictReader(stream) if r['kind']==kind and int(r['episode'])==sample['episode'] and int(r['frame'])==sample['frame'])
                old_error=max(abs(measure['h0_arm_raw']-float(old['h0_on_off_arm_raw_mean_abs'])),abs(measure['h0_gripper_raw']-float(old['h0_on_off_gripper_raw_abs'])))
                assert old_error<=1e-12,(kind,sample['id'],old_error)
            audits.append(dict(model=kind,checkpoint_step=step,probe_id=sample['id'],gate0_vs_physical_bypass_max=bypass_diff,
                current_vs_explicit_fp32_repeat_max=repeat_diff,r5_h0_reproduction_error=old_error,**measure))
            for dose in ['checkpoint',0.,.001,.01,.1,-.01]:
                output=on if dose=='checkpoint' else off if dose==0 else predict(policy,trace,batch,alpha=dose)
                raw=output*std+datasets['train'].stats['action']['mean'].numpy()
                doses.append(dict(model=kind,checkpoint_step=step,probe_id=sample['id'],episode=sample['episode'],frame=sample['frame'],
                    alpha=alpha if dose=='checkpoint' else dose,dose_label=str(dose),
                    units='normalized / raw OSC controller input',measurement='offline output relative to gate0',eval_dtype='float32',
                    diagnostic_only=True,ood_probe=dose not in ['checkpoint',0.],raw_output_exceeds_box=bool(np.abs(raw).max()>1),
                    **differences(output,off,std)))
        # Read-only gradient graphs use fixed Train/Dev batches and seed; no optimizer call.
        for split in ['train','dev']:
            items=[index[name] for name in lock['gradient_'+split+'_ids']]
            batch=datasets[split].batch([s['index'] for s in items])
            for alpha in [None,0.,.01]:grad_rows.extend(gradients(policy,trace,batch,kind,step,split,alpha))
        # Dependency controls; prototype permutation is an invariant of cosine-max pooling.
        original_bank=trace.oc.prototype_bank.prototypes.detach().clone()
        seed(1);permutation=np.random.default_rng(1).permutation(4).tolist()
        if permutation==list(range(4)):permutation=permutation[1:]+permutation[:1]
        reference=[s for s in samples if s['role']=='bank_reference'];vectors=[];random_sources=[]
        for s in reference:
            raw=datasets['train'].batch([s['index']])['_oc_raw_top']
            feats,_=trace.oc.dino(raw);choices=[i for i in range(256) if i not in [123,124]]
            chosen=int(np.random.default_rng(1+s['episode']).choice(choices))
            vectors.append(torch.nn.functional.normalize(feats[0,chosen],dim=0))
            random_sources.append(dict(episode=s['episode'],frame=s['frame'],patch=chosen))
        random_bank=torch.stack(vectors);replacement=datasets['train'].batch([s['index'] for s in samples if s['split']=='train' and s['role']!='bank_reference'][::-1][:1])
        for identity in lock['dependency_probe_ids']:
            s=index[identity];batch=datasets[s['split']].batch([s['index']])
            for gate in [None,.01]:
                real=predict(policy,trace,batch,alpha=gate)
                for variant in ['permuted_bank','random_train_patch_bank','object_top_only_swap','both_top_paths_swap']:
                    extra=None
                    try:
                        if variant=='permuted_bank':trace.oc.prototype_bank.prototypes.copy_(original_bank[permutation])
                        elif variant=='random_train_patch_bank':trace.oc.prototype_bank.prototypes.copy_(random_bank)
                        if variant=='object_top_only_swap':
                            replacement_top=replacement['observation.images.image'] if kind=='legacy' else replacement['_oc_raw_top']
                            extra=trace.oc.dino.register_forward_pre_hook(lambda _m,_args:(replacement_top,))
                        changed=dict(batch)
                        if variant=='both_top_paths_swap':
                            changed['observation.images.image']=replacement['observation.images.image'];changed['_oc_raw_top']=replacement['_oc_raw_top']
                        output=predict(policy,trace,changed,alpha=gate)
                        delta=differences(output,real,std)
                        if variant=='permuted_bank':assert delta['chunk_normalized_max']<=1e-6
                        dependencies.append(dict(model=kind,checkpoint_step=step,probe_id=identity,episode=s['episode'],frame=s['frame'],
                            alpha=float(trace.oc.gate) if gate is None else gate,variant=variant,
                            units='action difference versus real bank / unchanged top',measurement='offline input/bank counterfactual',eval_dtype='float32',
                            diagnostic_only=True,ood_probe=(gate is not None or variant!='permuted_bank'),semantic_unrelatedness_verified=False,
                            random_sources=json.dumps(random_sources) if variant=='random_train_patch_bank' else '',**delta))
                    finally:
                        if extra:extra.remove()
                        trace.oc.prototype_bank.prototypes.copy_(original_bank)
        assert model_hash(policy)==before,'Read-only probe changed parameter or buffer state'
        integrity.append(dict(model=kind,checkpoint_step=step,state_before_after_identical=True,checkpoint_sha256=source['sha256']))
        # Parameter/optimizer comparison is descriptive; the two checkpoints are not a full trajectory.
        named=dict(policy.named_parameters());ids=[]
        for group,live in zip(saved['optimizer']['param_groups'],policy.get_optim_params()):
            mapping={id(p):pid for p,pid in zip(live['params'],group['params'])};ids.append(mapping)
        lookup={k:v for m in ids for k,v in m.items()}
        for name,p in named.items():
            if name.startswith('model.object_conditioning.') and p.requires_grad:
                state=saved['optimizer']['state'].get(lookup[id(p)],{})
                comparison.append(dict(model=kind,checkpoint_step=step,parameter=name,parameter_l2=float(p.detach().norm()),
                    optimizer_step=float(state['step']) if state else None,
                    exp_avg_l2=float(state['exp_avg'].norm()) if state else None,
                    exp_avg_sq_l2=float(state['exp_avg_sq'].norm()) if state else None,
                    units='parameter / Adam moment norms',measurement='checkpoint state',eval_dtype='float32'))
        trace.close();del trace,policy,saved;gc.collect();torch.cuda.empty_cache()
    # Equivalent initial reconstruction, including full P1/DINO/bank state hash.
    _,policy=make_policy(kind);expected=json.loads((R5/'result/smoke'/kind/'validation.json').read_text())['initial_model_sha256']
    assert model_hash(policy)==expected,'INIT_GRAD_UNVERIFIED: initial hash mismatch'
    trace=Trace(policy)
    for split in ['train','dev']:
        items=[index[name] for name in lock['gradient_'+split+'_ids']]
        grad_rows.extend(gradients(policy,trace,datasets[split].batch([s['index'] for s in items]),kind,0,split))
    assert model_hash(policy)==expected
    trace.close();del trace,policy;gc.collect();torch.cuda.empty_cache()
    for name,rows in [('object_signal_scales',scales),('gate_dose_response',doses),('measurement_audit',audits),
                      ('gradients',grad_rows),('prototype_heatmap_index',heat_rows),('object_dependency_probes',dependencies),('checkpoint_5k_10k_comparison',comparison)]:
        csv_out(f'{name}_{kind}.csv',rows)
    write(OUT/f'heatmap_payload_{kind}.json',heat_data)
    write(OUT/f'probe_complete_{kind}.json',dict(status='D0_D4_PROBES_DONE',kind=kind,weights_saved=False,new_optimizer_updates=0,
        read_only_state_checks=integrity,initial_hash_verified=True,input_lock_sha256=sha(OUT/'input_lock.json')))
    print('D0_D4_DONE',kind,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['legacy','fixed'],required=True)
    parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    try:run(args.kind,args.pilot)
    except Exception:
        import traceback
        (OUT/f'error_{args.kind}.log').write_text(traceback.format_exc());raise
