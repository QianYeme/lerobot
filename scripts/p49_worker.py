"""Disposable gates/benchmarks and isolated resumable Gate+ runs."""
import argparse
import copy
import fcntl
import json
import os
import shutil
import time

from p49_runtime import (OUT, JOBS, P41, R5, baseline, batches, factory, restore_rng,
                         rng, row, sha, update, weight_hash, write)

def save(job, folder, policy, opt, old, step, epoch, cursor, identity):
    import torch
    path = folder / 'checkpoints' / f'{step:06d}' / 'resume.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial')
    torch.save(dict(model=policy.state_dict(), optimizer=opt.state_dict(), scheduler=None,
                    rng=rng(old, JOBS[job][0]), step=step, epoch=epoch, next_batch_cursor=cursor,
                    job=job, identity=identity), temp)
    temp.replace(path)
    receipt = dict(job=job, step=step, path=str(path), bytes=path.stat().st_size,
                   sha256=sha(path), complete_resume=True)
    row(folder / 'checkpoint_manifest.csv', receipt)
    write(folder / 'latest.json', receipt)
    return path

def main():
    p = argparse.ArgumentParser(); p.add_argument('--job', choices=list(JOBS), required=True)
    p.add_argument('--mode', choices=['smoke', 'bench', 'train'], required=True)
    p.add_argument('--tag', default='single'); args = p.parse_args()
    import torch
    job = args.job; domain, kind, seed, target = JOBS[job]
    folder = OUT / ('runs' if args.mode == 'train' else args.mode) / job
    if args.mode == 'bench': folder = folder / args.tag
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'worker.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old, cfg, policy, opt, data, pre, stats = factory(job, folder)
    source_checkpoint = baseline(job)
    gate = policy.model.object_conditioning.gate
    initial_non_gate = weight_hash(policy, True)
    identity = dict(job=job, domain=domain, kind=kind, seed=seed, target=target,
                    gate_initial_value=float(gate.detach()), initial_non_gate_sha256=initial_non_gate,
                    baseline_path=str(source_checkpoint), baseline_sha256=sha(source_checkpoint),
                    formal6_pairing='RECIPE_MATCHED_NOT_BYTE_MATCHED' if domain == 'f6' else 'BYTE_MATCHED_INIT',
                    stats_sha256=sha(P41/'train90_statistics.pt') if domain=='f6' else sha(R5/'result/train35_statistics.pt'),
                    bank_sha256=sha(P41/'formal6_train90_prototypes.pt') if domain=='f6' else sha(R5/'result/prototype_bank.pt'))
    write(folder / 'identity.json', identity)
    if args.mode == 'smoke':
        first_batch = next(iter(batches(job, data, pre, 0, 0)))
        initial_rng = rng(old, domain)
        initial_state = {k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
        grads = {}
        for label, alpha in [('plus', .001), ('zero', 0.)]:
            restore_rng(old, domain, initial_rng)
            with torch.no_grad(): gate.fill_(alpha)
            opt.zero_grad(set_to_none=True); policy.train()
            loss, _ = policy(first_batch); assert torch.isfinite(loss); loss.backward()
            grads[label] = {name:float(torch.sqrt(sum((v.grad.detach().double().square().sum() for n,v in policy.named_parameters()
                if token in n and v.grad is not None), torch.tensor(0.,device='cuda'))))
                for name, token in [('gate','object_conditioning.gate'),('adapter','object_conditioning.adapter'),
                                    ('attention','object_conditioning.cross_attention'),('slots','object_conditioning.token_slot_embedding')]}
        assert grads['zero']['adapter'] == grads['zero']['attention'] == 0
        assert grads['plus']['adapter'] > 0 and grads['plus']['attention'] > 0
        policy.load_state_dict(initial_state); restore_rng(old, domain, initial_rng)
        del initial_state
        # Reconstruct zero-initialized historical recipe, compare every other tensor.
        del policy, opt; torch.cuda.empty_cache()
        _, _, zero, _, _, _, _ = factory(job, folder/'zero_init_reference', 0.)
        zero_non_gate = weight_hash(zero, True)
        assert zero_non_gate == initial_non_gate
        if domain == 'lb':
            expected = json.loads((R5/f'result/smoke/{kind}/validation.json').read_text())['initial_model_sha256']
            assert old.model_hash(zero) == expected
        history = torch.load(source_checkpoint,map_location='cpu',weights_only=False)
        assert history['step'] == target
        zero.load_state_dict(history['model'], strict=True)
        assert torch.isfinite(zero.model.object_conditioning.gate)
        del zero, history; torch.cuda.empty_cache()
        old, cfg, policy, opt, data, pre, stats = factory(job, folder)
        if domain == 'lb': old.seed(1)
        iterator = iter(batches(job, data, pre, 0, 0))
        for step in range(1, 11): update(job, policy, opt, next(iterator))
        path = save(job, folder, policy, opt, old, 10, 0, 10, identity)
        next_batch = next(iterator)
        expected_loss = update(job, policy, opt, next_batch)
        expected_hash = weight_hash(policy)
        state = torch.load(path,map_location='cpu',weights_only=False)
        policy.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        restore_rng(old, domain, state['rng'])
        actual_loss = update(job, policy, opt, next_batch)
        resume_identical = expected_hash == weight_hash(policy)
        assert resume_identical, (expected_loss, actual_loss)
        # Recreate loader at saved cursor; no update on this equality probe.
        replay = next(iter(batches(job, data, pre, 0, 10)))
        assert all(torch.equal(next_batch[k], replay[k]) for k in next_batch if torch.is_tensor(next_batch[k]))
        write(folder/'validation.json',dict(status='PASS',identity=identity,initial_non_gate_equal=True,
             gradients=grads,baseline_strict_load=True,exact_next_update_hash=resume_identical,
             sampler_cursor_batch_equal=True,smoke_updates_discarded=True,formal_updates=0))
        return
    if domain == 'lb': old.seed(1)
    step=epoch=cursor=0
    if args.mode == 'train' and (folder/'latest.json').exists():
        latest=json.loads((folder/'latest.json').read_text());assert sha(latest['path'])==latest['sha256']
        state=torch.load(latest['path'],map_location='cpu',weights_only=False)
        assert state['job']==job and state['identity']==identity
        policy.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        restore_rng(old,domain,state['rng']);step=state['step'];epoch=state['epoch'];cursor=state['next_batch_cursor'];del state
    target = 100 if args.mode=='bench' else target
    started=time.time();timed=None;initial_step=step
    deadline=json.loads((OUT/'job_registry.json').read_text())['training_deadline']
    while step < target:
        for batch in batches(job,data,pre,epoch,cursor):
            if args.mode=='bench' and step==20:torch.cuda.synchronize();timed=time.time()
            loss,grad=update(job,policy,opt,batch);step+=1;cursor+=1
            if args.mode=='bench' or step%10==0:
                row(folder/'training_metrics.csv',dict(step=step,epoch=epoch,cursor=cursor,loss=loss,gradient_norm=grad,
                    gate=float(gate.detach()),elapsed_s=time.time()-started))
            if args.mode=='train' and step%100==0:
                write(folder/'heartbeat.json',dict(job=job,pid=os.getpid(),step=step,target=target,
                    updates_per_second=(step-initial_step)/(time.time()-started),time=time.time()))
            save_points=[1000,5000,10000] if domain=='lb' else [10000,20000,40000,60000,80000]
            stopping=args.mode=='train' and (time.time()>=deadline or shutil.disk_usage(OUT).free<10*1024**3)
            if args.mode=='train' and (step in save_points or stopping):save(job,folder,policy,opt,old,step,epoch,cursor,identity)
            if step>=target or stopping:break
        else:epoch+=1;cursor=0;continue
        break
    torch.cuda.synchronize()
    if args.mode=='bench':
        write(folder/'benchmark.json',dict(job=job,status='DONE',timed_updates=80,
            timed_seconds=time.time()-timed,updates_per_second=80/(time.time()-timed),
            peak_memory_bytes=torch.cuda.max_memory_allocated(),final_sha256=weight_hash(policy),discarded=True))
    else:
        write(folder/'training_complete.json',dict(job=job,status='DONE' if step==target else 'PARTIAL',
            step=step,target=target,elapsed_s=time.time()-started,identity=identity))

if __name__=='__main__':main()
