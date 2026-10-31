"""P4.9 pair-specific factories; historical recipes remain read-only."""
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys

ROOT = Path('/root/autodl-tmp/lerobot-main')
WORK = ROOT / 'experiments/p4_final_p1_training/p4_9_unified_gate_init/index1'
OUT = WORK / 'result'
R5 = ROOT / 'experiments/p4_public_benchmark/p4_8_libero_pilot/index5'
P41 = ROOT / 'experiments/p4_final_p1_training/p4_1_unattended_p1_pilot/index3/result'
FIX = ROOT / 'experiments/p4_final_p1_training/p4_2_1_dino_repair_parallel/index1'
JOBS = {
    'F6_LEGACY_GATEPLUS_S1': ('f6', 'legacy', 1, 80000),
    'F6_FIXED_GATEPLUS_S3': ('f6', 'fixed', 3, 80000),
    'LB_LEGACY_GATEPLUS': ('lb', 'legacy', 1, 10000),
    'LB_FIXED_GATEPLUS': ('lb', 'fixed', 1, 10000),
}

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def row(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(value))
        if stream.tell() == 0: w.writeheader()
        w.writerow(value)

def backend(job):
    domain, kind, seed, target = JOBS[job]
    if domain == 'lb':
        sys.path.insert(0, str(R5 / 'workspace/scripts'))
        import p48_r5_common as old
        return old
    sys.path.insert(0, str(WORK / 'workspace' / ('f6_fixed' if kind == 'fixed' else 'f6_legacy') / 'src'))
    sys.path.insert(0, str(WORK / 'workspace/f6_helpers'))
    import p4_1_pilot as old
    import torch
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)
    return old

def factory(job, folder, gate_init=.001):
    import numpy as np
    import torch
    old = backend(job)
    domain, kind, seed, target = JOBS[job]
    folder.mkdir(parents=True, exist_ok=True)
    if domain == 'lb':
        cfg, policy = old.make_policy(kind)
        with torch.no_grad(): policy.model.object_conditioning.gate.fill_(gate_init)
        data = old.CachedData()
        opt = cfg.get_optimizer_preset().build(policy.get_optim_params())
        pre = None; stats = data.stats
    else:
        from p4_2_1_repair import RawTopProcessor
        old.OUT = folder
        for name in ['formal6_train90_prototypes.pt', 'prototype_provenance.json', 'train90_statistics.pt']:
            if not (folder / name).exists(): shutil.copyfile(P41 / name, folder / name)
        cfg = old.config()
        cfg.object_conditioning_gate_init = gate_init
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        ids = json.loads((old.PRIOR / 'train90_episode_ids.json').read_text())
        data = old.Mapped(ids)
        stats = torch.load(P41 / 'train90_statistics.pt', map_location='cpu', weights_only=False)
        standard, _ = old.make_act_pre_post_processors(cfg, stats)
        # Legacy consumes the original normalized Top; Fixed consumes preserved raw Top.
        pre = RawTopProcessor(standard, cfg.device, legacy=kind == 'legacy') if kind == 'fixed' else standard
        policy = old.ACTPolicy(cfg).cuda()
        opt = torch.optim.AdamW(policy.get_optim_params(), lr=1e-5, weight_decay=1e-4)
    assert not any(p.requires_grad for p in policy.model.object_conditioning.dino.parameters())
    assert policy.model.object_conditioning.gate.requires_grad
    cfg.save_pretrained(folder / 'effective_model_config')
    return old, cfg, policy, opt, data, pre, stats

def batches(job, data, pre, epoch, cursor):
    domain, kind, seed, target = JOBS[job]
    if domain == 'lb':
        import p48_r5_common as old
        for indices in old.epoch_order(len(data), epoch)[cursor:]: yield data.batch(indices)
    else:
        from p4_2_queue import batches as original_batches
        for raw in original_batches(data, seed, epoch, cursor): yield pre(raw)

def weight_hash(policy, exclude_gate=False):
    h = hashlib.sha256()
    for name, value in policy.state_dict().items():
        if exclude_gate and name == 'model.object_conditioning.gate': continue
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def rng(old, domain):
    return old.rng()

def restore_rng(old, domain, value):
    (old.restore_rng if domain == 'lb' else old.restore)(value)

def update(job, policy, opt, batch):
    domain = JOBS[job][0]
    if domain == 'lb':
        import p48_r5_common as old
        loss, grad, _ = old.update(policy, opt, batch, policy.config.get_optimizer_preset().grad_clip_norm)
        return loss, grad
    from p4_2_queue import update as original_update
    return original_update(policy, opt, batch)

def baseline(job):
    domain, kind, seed, target = JOBS[job]
    if domain == 'lb': return R5 / f'result/runs/{kind}/checkpoints/010000/resume.pt'
    if kind == 'legacy': return P41 / 'checkpoints/pilot/080000/resume.pt'
    return FIX / 'result/workers/F3/runs/F3/checkpoints/formal/080000/resume.pt'
