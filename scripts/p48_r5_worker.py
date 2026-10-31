"""Isolated smoke, disposable throughput probes and resumable 10k r5 workers."""
import argparse
import csv
import fcntl
import json
import shutil
import time

from p48_r5_common import OUT,CachedData,config,epoch_order,make_policy,model_hash,restore_rng,rng,seed,sha,update,write


def effective(kind,cfg,policy,preset):
    folder=OUT/'effective_configs'/kind
    cfg.save_pretrained(folder)
    # Export true class defaults independently; don't use a manually edited effective instance.
    from lerobot.policies.act.configuration_act import ACTConfig
    ACTConfig().save_pretrained(OUT/'effective_configs/native_class_default')
    native=json.loads((folder/'config.json').read_text())
    record=dict(kind=kind,act_config=native,optimizer=vars(preset),seed=1,steps=10000,
        batch_size=8,drop_last=False,augmentation=False,amp=False,train_rgb_flip=False,
        train_stats_sha256=sha(OUT/'train35_statistics.pt'),split_sha256=sha(OUT/'dataset_split_manifest.json'),
        sampler='numpy PCG64 SeedSequence([1,epoch]); exact epoch/cursor checkpointed',
        numerical_mode='float32; deterministic algorithms; TF32 off; CUBLAS_WORKSPACE_CONFIG=:4096:8',
        steps_and_seed_and_cache_sampler_are_explicit_training_loop_overrides=True,
        original_stock_commit='8fff0fde7c79f23a93d845d1a50e985de01f8b8a',
        object_conditioning=None if kind=='act' else dict(prototype_bank_sha256=sha(OUT/'prototype_bank.pt'),
            dino='frozen dinov2_vits14 Top only',prototype_normalizations=1,
            online_dino_normalizations=2 if kind=='legacy' else 1,
            prototypes=4,object_tokens=5,local_regions='fixed2x2',temperature=.20,
            raw_token_dim=387,adapter=[387,256,512],cross_attention='ACT encoder input Q, object K/V',
            gate='unbounded signed scalar, init0',fusion='before native ACT transformer encoder'))
    write(OUT/'effective_configs'/f'{kind}.yaml',record)  # JSON syntax is also valid YAML.


def checkpoint(folder,policy,opt,kind,step,epoch,cursor):
    import torch
    path=folder/'checkpoints'/f'{step:06d}'/'resume.pt';path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.partial')
    torch.save(dict(model=policy.state_dict(),optimizer=opt.state_dict(),scheduler=None,rng=rng(),kind=kind,
        step=step,epoch=epoch,next_batch_cursor=cursor,split_sha256=sha(OUT/'dataset_split_manifest.json'),
        bank_sha256=sha(OUT/'prototype_bank.pt') if kind!='act' else None,
        stats_sha256=sha(OUT/'train35_statistics.pt')),temp)
    temp.replace(path)
    result=dict(kind=kind,step=step,path=str(path),sha256=sha(path),complete_resume=True)
    write(folder/'latest.json',result)
    with (folder/'checkpoint_manifest.csv').open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result))
        if f.tell()==0:w.writeheader()
        w.writerow(result)
    return path


def load_checkpoint(path,kind,policy,opt):
    import torch
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['kind']==kind and saved['split_sha256']==sha(OUT/'dataset_split_manifest.json')
    if kind!='act':assert saved['bank_sha256']==sha(OUT/'prototype_bank.pt')
    assert saved['stats_sha256']==sha(OUT/'train35_statistics.pt')
    policy.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
    return saved


def smoke(kind,folder,data,cfg,policy,opt,preset):
    import torch
    import torch.nn.functional as F
    batch=data.batch(list(range(8)))
    cnn=[]
    def capture_cnn(_module,args,output):
        import hashlib
        cnn.append(dict(input_sha256=hashlib.sha256(args[0].detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                        feature_sha256=hashlib.sha256(output['feature_map'].detach().cpu().contiguous().numpy().tobytes()).hexdigest()))
    handle=policy.model.backbone.register_forward_hook(capture_cnn)
    initial_common=model_hash(policy,True);initial_full=model_hash(policy)
    policy.eval()
    actual_dino=[]
    if kind!='act':
        encoder=policy.model.object_conditioning.dino.model;original=encoder.forward_features
        def capture_dino(x):actual_dino.append(x.detach().clone());return original(x)
        encoder.forward_features=capture_dino
    with torch.inference_mode():chunk=policy.predict_action_chunk(batch)
    handle.remove();assert chunk.shape==(8,100,7) and torch.isfinite(chunk).all()
    if kind!='act':
        encoder.forward_features=original
        selected=batch['observation.images.image'] if kind=='legacy' else batch['_oc_raw_top']
        mean=selected.new_tensor([.485,.456,.406]).view(1,3,1,1);std=selected.new_tensor([.229,.224,.225]).view(1,3,1,1)
        expected=(F.interpolate(selected,(224,224),mode='bilinear',align_corners=False)-mean)/std
        assert len(actual_dino)==1 and torch.equal(actual_dino[0],expected)
        assert policy.model.object_conditioning.last_debug_shapes['object_tokens']==(8,5,387)
    seed(1);started=time.time()
    with (folder/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['step','loss','gradient_norm']);w.writeheader()
        for step in range(1,101):
            loss,grad,_=update(policy,opt,batch,preset.grad_clip_norm)
            w.writerow(dict(step=step,loss=loss,gradient_norm=grad))
    path=checkpoint(folder,policy,opt,kind,100,0,0)
    first=update(policy,opt,batch,preset.grad_clip_norm)
    expected={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
    del policy,opt;torch.cuda.empty_cache()
    _,policy=make_policy(kind);opt=preset.build(policy.get_optim_params());load_checkpoint(path,kind,policy,opt)
    second=update(policy,opt,batch,preset.grad_clip_norm)
    difference=max(float((expected[k]-v.detach().cpu()).abs().max()) for k,v in policy.state_dict().items())
    assert difference<=1e-6,(kind,difference,first[:2],second[:2])
    report=dict(status='REAL_TRAIN_100_UPDATE_RESTORE_PASS',kind=kind,discarded_updates=True,
        formal_updates=0,initial_common_sha256=initial_common,initial_model_sha256=initial_full,
        initial_cnn_inputs_and_features=cnn,next_update_max_difference=difference,
        dino_effective_input_verified=kind!='act',train_rgb_flip=False,
        output_shape=list(chunk.shape),elapsed_s=time.time()-started,
        params=sum(p.numel() for p in policy.parameters()),trainable_params=sum(p.numel() for p in policy.parameters() if p.requires_grad),
        peak_memory_bytes=torch.cuda.max_memory_allocated(),checkpoint=str(path),checkpoint_sha256=sha(path))
    write(folder/'validation.json',report);print(json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['act','legacy','fixed'],required=True)
    parser.add_argument('--mode',choices=['smoke','bench','train'],required=True);parser.add_argument('--tag',default='single')
    args=parser.parse_args()
    import torch
    folder=OUT/('runs' if args.mode=='train' else args.mode)/args.kind
    if args.mode=='bench':folder=folder/args.tag
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.mode=='train' and args.kind!='act':
        assert json.loads((OUT/'prototype_gate.json').read_text())['status']=='G2_PROTOTYPE_PASS'
    data=CachedData();cfg,policy=make_policy(args.kind)
    preset=cfg.get_optimizer_preset();opt=preset.build(policy.get_optim_params())
    effective(args.kind,cfg,policy,preset)
    if args.mode=='smoke':smoke(args.kind,folder,data,cfg,policy,opt,preset);return
    initial_common=model_hash(policy,True);initial_full=model_hash(policy);seed(1)
    if args.mode=='train':
        assert initial_common==json.loads((OUT/'smoke'/args.kind/'validation.json').read_text())['initial_common_sha256']
    step=0;epoch=0;cursor=0
    if args.mode=='train' and (folder/'latest.json').exists():
        saved=load_checkpoint(json.loads((folder/'latest.json').read_text())['path'],args.kind,policy,opt)
        step=saved['step'];epoch=saved['epoch'];cursor=saved['next_batch_cursor']
    if step==10000:return
    target=100 if args.mode=='bench' else 10000
    started=time.time();timed=None;first_step=step
    history=folder/('training_metrics.csv' if args.mode=='train' else 'metrics.csv')
    fields=['step','epoch','cursor','loss','gradient_norm','elapsed_s']
    with history.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields)
        if f.tell()==0:w.writeheader()
        while step<target:
            order=epoch_order(len(data),epoch)
            while cursor<len(order) and step<target:
                if args.mode=='bench' and step==20:torch.cuda.synchronize();timed=time.time()
                batch=data.batch(order[cursor]);loss,grad,_=update(policy,opt,batch,preset.grad_clip_norm)
                step+=1;cursor+=1
                if args.mode=='bench' or step%10==0:
                    w.writerow(dict(step=step,epoch=epoch,cursor=cursor,loss=loss,gradient_norm=grad,elapsed_s=time.time()-started));f.flush()
                if args.mode=='train' and step%100==0:
                    write(folder/'heartbeat.json',dict(kind=args.kind,step=step,target=target,epoch=epoch,cursor=cursor,
                        elapsed_s=time.time()-started,updates_per_second=(step-first_step)/(time.time()-started),timestamp=time.time()))
                    if shutil.disk_usage(OUT).free<8*1024**3:
                        checkpoint(folder,policy,opt,args.kind,step,epoch,cursor)
                        raise RuntimeError('Resource gate: free disk below8GiB; recoverable checkpoint retained')
                if args.mode=='train' and step in [5000,10000]:checkpoint(folder,policy,opt,args.kind,step,epoch,cursor)
            if cursor==len(order):epoch+=1;cursor=0
    torch.cuda.synchronize();finished=time.time()
    if args.mode=='bench':
        report=dict(status='DISPOSABLE_BENCH_DONE',kind=args.kind,updates=100,timed_updates=80,
            timed_wall_seconds=finished-timed,updates_per_second=80/(finished-timed),
            total_wall_seconds=finished-started,initial_common_sha256=initial_common,initial_model_sha256=initial_full,
            final_model_sha256=model_hash(policy),peak_memory_bytes=torch.cuda.max_memory_allocated(),discarded_updates=True)
        write(folder/'benchmark.json',report)
    else:
        policy.eval();one=data.batch([0]);latencies=[]
        with torch.inference_mode():
            for i in range(30):
                torch.cuda.synchronize();t=time.perf_counter();policy.predict_action_chunk(one);torch.cuda.synchronize()
                if i>=5:latencies.append(time.perf_counter()-t)
        report=dict(status='TRAIN_10000_COMPLETE',kind=args.kind,updates=step,epoch=epoch,next_cursor=cursor,
            elapsed_s=finished-started,updates_this_process=step-first_step,initial_model_sha256=initial_full,
            params=sum(p.numel() for p in policy.parameters()),trainable_params=sum(p.numel() for p in policy.parameters() if p.requires_grad),
            peak_memory_bytes=torch.cuda.max_memory_allocated(),batch1_chunk_inference_seconds=latencies,
            source_split_sha256=sha(OUT/'dataset_split_manifest.json'),prototype_bank_sha256=sha(OUT/'prototype_bank.pt') if args.kind!='act' else None,
            pilot_only=True,latency_scope='batch1 chunk100 prediction; not per-action control period')
        write(folder/'training_complete.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
