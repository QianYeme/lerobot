"""One fresh retry of failed P4.2 tasks, isolated decoder state and unchanged recipe."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import p4_1_pilot as old
import p4_2_queue as queue
from lerobot.datasets import video_utils

ROOT=old.ROOT
P42=ROOT/'experiments/p4_final_p1_training/p4_2_unattended_multi_model/index1/result'
OUT=P42/'retry1'


def clear_decoder_cache(_worker_id):
    # File handles opened by the parent must not share seek offsets across forked readers.
    video_utils._default_decoder_cache.clear()


def safe_batches(dataset,seed,epoch,cursor):
    indices=np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(len(dataset)).tolist()
    indices=indices[:len(indices)//4*4][cursor*4:]
    return DataLoader(Subset(dataset,indices),batch_size=4,num_workers=4,shuffle=False,
                      generator=torch.Generator().manual_seed(seed+epoch),pin_memory=True,
                      worker_init_fn=clear_decoder_cache)


def worker(name,seed,p1,cutoff):
    queue.OUT=OUT/name
    queue.OUT.mkdir(parents=True,exist_ok=False)
    queue.batches=safe_batches
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(False)
    train=json.loads((old.PRIOR/'train90_episode_ids.json').read_text())
    dev=json.loads((old.PRIOR/'dev15_episode_ids.json').read_text())
    (queue.OUT/'attempt_identity.json').write_text(json.dumps(dict(attempt=1,start_step=0,
        semantics='fresh engineering retry; failed attempt discarded, not exact resume',
        reason='TorchCodec native decoder handles must not be inherited across fork',
        recipe_changed=False,seed=seed,model='P1_LEGACY_DOUBLE_NORM' if p1 else 'Original_ACT_8D'),indent=2))
    queue.run(name,seed,p1,False,train,dev,cutoff)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker')
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--p1',type=int,default=1)
    parser.add_argument('--cutoff',type=float,default=0)
    args=parser.parse_args()
    if args.worker:
        worker(args.worker,args.seed,bool(args.p1),args.cutoff)
        return
    OUT.mkdir(exist_ok=False)
    (OUT/'controller_identity.json').write_text(json.dumps(dict(pid=os.getpid(),started_unix=time.time())))
    started=json.loads((P42/'queue_started.json').read_text())['start_unix']
    remaining=max(0,8*3600-(time.time()-started))
    cutoff=time.monotonic()+remaining
    tasks=[('ACT_S1_retry1',1,0,'ACT1'),('P1_S2_retry1',2,1,'L2'),('P1_S3_retry1',3,1,'L3')]
    registry={label:str(OUT/name/'runs'/name/'checkpoints/formal/080000/resume.pt') for name,_,_,label in tasks}
    (P42/'retry_checkpoint_registry.json').write_text(json.dumps(registry,indent=2))
    for name,seed,p1,label in tasks:
        if time.monotonic()>=cutoff:break
        with (OUT/(name+'.log')).open('w') as log:
            process=subprocess.Popen([sys.executable,__file__,'--worker',name,'--seed',str(seed),'--p1',str(p1),
                '--cutoff',str(cutoff)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            (OUT/'active.json').write_text(json.dumps(dict(run=name,pid=process.pid,start_unix=time.time(),cutoff_monotonic=cutoff)))
            code=process.wait()
        with (OUT/'execution.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(run=name,returncode=code,finished_unix=time.time()))+'\n')
    (OUT/'controller_complete.json').write_text(json.dumps(dict(done=True,finished_unix=time.time())))


if __name__=='__main__':
    main()
