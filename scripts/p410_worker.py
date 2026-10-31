"""Frozen r5 recipe, four authorized learnable-gate initializations, 5k each."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path('/root/autodl-tmp/lerobot-main')
R5=ROOT/'experiments/p4_public_benchmark/p4_8_libero_pilot/index5'
WORK=ROOT/'experiments/p4_public_benchmark/p4_10_gate_sweep/index1'
OUT=WORK/'result'
sys.path.insert(0,str(R5/'workspace/scripts'))
import p48_r5_common as old
VALUES={'G1e-2':.01,'G5e-2':.05,'G1e-1':.1,'GNEG':-.01}

def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=VALUES,required=True)
    p.add_argument('--mode',choices=['smoke','bench','train'],required=True);p.add_argument('--tag',default='single');a=p.parse_args()
    import torch
    folder=OUT/a.mode/a.variant/(a.tag if a.mode=='bench' else '')
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg,policy=old.make_policy('fixed');original=old.model_hash(policy)
    cfg.save_pretrained(folder/'effective_model_config')
    expected=json.loads((R5/'result/smoke/fixed/validation.json').read_text())['initial_model_sha256'];assert original==expected
    with torch.no_grad():policy.model.object_conditioning.gate.fill_(VALUES[a.variant])
    opt=cfg.get_optimizer_preset().build(policy.get_optim_params());data=old.CachedData();old.seed(1)
    identity=dict(variant=a.variant,gate_initial=VALUES[a.variant],seed=1,budget=5000,zero_init_sha=original,
        initial_sha=old.model_hash(policy),bank_sha=old.sha(R5/'result/prototype_bank.pt'),
        stats_sha=old.sha(R5/'result/train35_statistics.pt'),split_sha=old.sha(R5/'result/dataset_split_manifest.json'),
        source_sha=old.sha(Path(__file__)),kind='fixed',sampler='r5 epoch_order SeedSequence([1,epoch]), batch8',
        only_scientific_change='learnable scalar gate initialization')
    old.write(folder/'identity.json',identity)
    import p48_r5_worker as historical
    historical.OUT=R5/'result'
    step=epoch=cursor=0;started=time.time();timed=None
    def save():
        return historical.checkpoint(folder,policy,opt,'fixed',step,epoch,cursor)
    if a.mode=='train' and (folder/'latest.json').exists():
        latest=json.loads((folder/'latest.json').read_text());assert old.sha(latest['path'])==latest['sha256']
        state=historical.load_checkpoint(latest['path'],'fixed',policy,opt)
        step=state['step'];epoch=state['epoch'];cursor=state['next_batch_cursor'];del state
    target=20 if a.mode=='smoke' else 100 if a.mode=='bench' else 5000
    with (folder/'train_metrics.csv').open('a',newline='') as stream:
        fields=['step','loss','gradient_norm_preclip','gate','gate_gradient_postclip','adapter_gradient_postclip','attention_gradient_postclip','elapsed_s']
        writer=csv.DictWriter(stream,fieldnames=fields)
        if stream.tell()==0:writer.writeheader()
        while step<target:
            order=old.epoch_order(len(data),epoch)
            for indices in order[cursor:]:
                if a.mode=='bench' and step==20:torch.cuda.synchronize();timed=time.time()
                batch=data.batch(indices);loss,grad,_=old.update(policy,opt,batch,cfg.get_optimizer_preset().grad_clip_norm)
                step+=1;cursor+=1
                if step%10==0:
                    def norm(token):return sum(float(v.grad.detach().double().square().sum()) for n,v in policy.named_parameters() if token in n and v.grad is not None)**.5
                    writer.writerow(dict(step=step,loss=loss,gradient_norm_preclip=grad,gate=float(policy.model.object_conditioning.gate.detach()),
                        gate_gradient_postclip=float(policy.model.object_conditioning.gate.grad),
                        adapter_gradient_postclip=norm('object_conditioning.adapter'),attention_gradient_postclip=norm('object_conditioning.cross_attention'),elapsed_s=time.time()-started));stream.flush()
                if a.mode=='train' and step%100==0:old.write(folder/'heartbeat.json',dict(step=step,target=target,pid=os.getpid(),elapsed_s=time.time()-started))
                if a.mode=='train' and step in [1000,5000]:save()
                if step==target:break
            if cursor==len(order):epoch+=1;cursor=0
    torch.cuda.synchronize()
    report=dict(status='DONE',mode=a.mode,variant=a.variant,steps=step,discarded=a.mode!='train',elapsed_s=time.time()-started,
        final_sha=old.model_hash(policy),peak_memory_bytes=torch.cuda.max_memory_allocated())
    if a.mode=='bench':report.update(timed_updates=80,timed_seconds=time.time()-timed)
    if a.mode=='smoke':
        path=save();batch=data.batch(old.epoch_order(len(data),epoch)[cursor])
        expected_loss=old.update(policy,opt,batch,cfg.get_optimizer_preset().grad_clip_norm);expected_hash=old.model_hash(policy)
        historical.load_checkpoint(path,'fixed',policy,opt)
        old.update(policy,opt,batch,cfg.get_optimizer_preset().grad_clip_norm)
        assert old.model_hash(policy)==expected_hash
        report['exact_next_update_restore']=True
    old.write(folder/'complete.json',report)

if __name__=='__main__':main()
