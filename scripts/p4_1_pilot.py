"""Frozen P1 formal6 pilot: read-only mapping, bounded single-seed queue, Dev-only evidence."""
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.object_conditioning.modeling_object_conditioning import FrozenDINOv2Small
from lerobot.policies.act.processor_act import make_act_pre_post_processors

ROOT = Path('/root/autodl-tmp/lerobot-main')
WORK = ROOT / 'experiments/p4_final_p1_training/p4_1_unattended_p1_pilot/index3'
OUT = WORK / 'result'
PRIOR = ROOT / 'experiments/p4_final_p1_training/p4_0_1_training_readiness_unblock/index1/result'
DATA = ROOT / 'outputs/p4_0/formal6_merged'
FIELDS = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos',
          'wrist_roll.pos', 'gripper.pos', 'gripper.load', 'gripper.curr']
REV = '83a829af6df245ed53596dbc22b18c19f8bd3315'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def row(name, value):
    path = OUT / name
    exists = path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(value))
        if not exists:
            writer.writeheader()
        writer.writerow(value)


def seed():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)


class Mapped(Dataset):
    def __init__(self, ids, chunks=True):
        self.ids = list(ids)
        self.base = LeRobotDataset('QYyyyyyyy/formal6_merged', root=DATA, episodes=self.ids,
                                  delta_timestamps={'action': [i / 30 for i in range(100)]} if chunks else None,
                                  video_backend='torchcodec')
        names = self.base.meta.features['observation.state']['names']
        self.indices = [names.index(name) for name in FIELDS]
        assert len(set(self.indices)) == 8 and 'master_gripper.pos' not in FIELDS
        assert self.base.meta.features['action']['names'] == FIELDS[:6]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        item['observation.state'] = item['observation.state'][self.indices]
        return {k: item[k] for k in ('observation.state', 'action', 'action_is_pad',
                                     'observation.images.top', 'observation.images.gripper') if k in item}


def statistics(ids):
    frame = pd.read_parquet(DATA / 'data/chunk-000/file-000.parquet',
                            filters=[('episode_index', 'in', ids)])
    assert set(frame.episode_index) == set(ids)
    names = json.loads((DATA / 'meta/info.json').read_text())['features']['observation.state']['names']
    stats = {}
    for key in ('observation.state', 'action'):
        values = np.stack(frame[key])
        if key == 'observation.state':
            values = values[:, [names.index(name) for name in FIELDS]]
        assert np.isfinite(values).all()
        stats[key] = {'mean': torch.tensor(values.mean(0)), 'std': torch.tensor(values.std(0).clip(1e-6))}
    for camera in ('top', 'gripper'):
        stats['observation.images.' + camera] = {
            'mean': torch.tensor([.485, .456, .406]).view(3, 1, 1),
            'std': torch.tensor([.229, .224, .225]).view(3, 1, 1)}
    torch.save(stats, OUT / 'train90_statistics.pt')
    return stats


def prototypes(ids):
    # Fixed Train-only quarter-order sources, frame 25%; reviewed visible cup interiors.
    sources = [(0, 194, (550, 325)), (29, 194, (448, 279)),
               (58, 194, (421, 251)), (86, 194, (505, 300))]
    assert all(ep in ids for ep, _, _ in sources)
    import av
    meta = pd.read_parquet(DATA / 'meta/episodes/chunk-000/file-000.parquet')
    encoder = FrozenDINOv2Small('dinov2_vits14').cuda().eval()
    bank, provenance = [], []
    for ep, frame, center in sources:
        m = meta[meta.episode_index == ep].iloc[0]
        key = 'videos/observation.images.top/'
        path = DATA / 'videos/observation.images.top' / f'chunk-{int(m[key+"chunk_index"]):03d}' / f'file-{int(m[key+"file_index"]):03d}.mp4'
        target = round(m[key + 'from_timestamp'] * 30) + frame
        with av.open(str(path)) as container:
            image = next(f.to_ndarray(format='rgb24') for i, f in enumerate(container.decode(video=0)) if i == target)
        pixels = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255)[None].cuda()
        patches, _ = encoder(pixels)
        # Inner cup neighborhood; mask is a prototype sampling region, not segmentation GT.
        yy, xx = torch.meshgrid(torch.arange(16, device='cuda'), torch.arange(16, device='cuda'), indexing='ij')
        mask = (((xx + .5) * 640 / 16 - center[0]) / 35).square() + (((yy + .5) * 480 / 16 - center[1]) / 35).square() <= 1
        assert mask.sum() > 0
        bank.append(F.normalize(patches[0, mask.flatten()].mean(0), dim=0).cpu())
        provenance.append(dict(episode=ep, frame=frame, center_xy=center, pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                               sampling='visible cup inner ellipse radius35px, executor reviewed Train images'))
    torch.save({'prototypes': torch.stack(bank), 'provenance': provenance}, OUT / 'formal6_train90_prototypes.pt')
    write('prototype_provenance.json', provenance)
    del encoder
    torch.cuda.empty_cache()


def config():
    folder = OUT / 'initial_config'
    folder.mkdir(exist_ok=True)
    values = json.loads((ROOT / 'experiments/position/d2_1_evidence_completion/index1/result/B3_config.json').read_text())
    values['object_conditioning_prototype_path'] = str(OUT / 'formal6_train90_prototypes.pt')
    (folder / 'config.json').write_text(json.dumps(values))
    return PreTrainedConfig.from_pretrained(folder)


def rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore(value):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['cpu'])
    torch.cuda.set_rng_state_all(value['cuda'])


def checkpoint(policy, optimizer, step, epoch, cursor, cfg, phase):
    folder = OUT / 'checkpoints' / phase / f'{step:06d}'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'resume.pt'
    temp = path.with_suffix('.partial')
    torch.save(dict(model=policy.state_dict(), optimizer=optimizer.state_dict(), scheduler=None,
                    step=step, epoch=epoch, next_batch_cursor=cursor, rng=rng(), config=cfg.to_dict() if hasattr(cfg, 'to_dict') else str(cfg),
                    dataset_revision=REV, split_sha=sha(PRIOR / 'split_freeze_manifest.json'),
                    prototypes_sha=sha(OUT / 'formal6_train90_prototypes.pt')), temp)
    temp.replace(path)
    cfg.save_pretrained(folder / 'pretrained_model')
    row('checkpoint_manifest.csv', dict(phase=phase, step=step, path=str(path), sha256=sha(path), complete_resume=True))
    return path


def loader(dataset, epoch, cursor=0):
    indices = np.random.default_rng(np.random.SeedSequence([1, epoch])).permutation(len(dataset)).tolist()
    indices = indices[:len(indices) // 4 * 4][cursor * 4:]
    return DataLoader(Subset(dataset, indices), batch_size=4, num_workers=4, shuffle=False,
                      generator=torch.Generator().manual_seed(1 + epoch), pin_memory=True)


def evaluate(policy, pre, stats, ids, step):
    dataset = Mapped(ids)
    meta = pd.read_parquet(DATA / 'meta/episodes/chunk-000/file-000.parquet')
    selected, offset = [], 0
    for ep in ids:
        length = int(meta[meta.episode_index == ep].iloc[0].length)
        selected.extend((ep, offset + min(length - 1, int(q * length)), q) for q in (0, .15, .3, .5, .7, .9))
        offset += length
    # Dataset native filtering order is ascending episode ID; frozen manifests are sorted.
    assert ids == sorted(ids)
    gate = policy.model.object_conditioning.gate
    trace = {}
    def audit_attention(_module, args, output):
        trace['object_residual_norm'] = float((gate.detach() * output[0].detach()).norm())
        trace['object_query_norm'] = float(args[0].detach().norm())
    handle = policy.model.object_conditioning.cross_attention.register_forward_hook(audit_attention)
    for ep, index, q in selected:
        raw = dataset[index]
        batch = pre(raw)
        state = raw['observation.state'].cuda()
        with torch.inference_mode():
            pred = policy.predict_action_chunk(batch)
            on_trace = dict(trace)
            saved = gate.detach().clone()
            gate.zero_()
            off = policy.predict_action_chunk(batch)
            gate.copy_(saved)
        mask = ~batch['action_is_pad'][0]
        error = (pred[0] - batch['action'][0]).abs()[mask]
        physical = error * stats['action']['std'].cuda()
        actual = pred[0, 0] * stats['action']['std'].cuda() + stats['action']['mean'].cuda()
        row('dev15_checkpoint_comparison.csv', dict(step=step, episode=ep, proxy_fraction=q,
            arm_normalized=float(error[:, :5].mean()), gripper_normalized=float(error[:, 5].mean()),
            arm_raw_mae=float(physical[:, :5].mean()), gripper_raw_mae=float(physical[:, 5].mean()),
            shoulder_h0_delta=float(actual[0] - state[0]), object_on_off_mean_abs=float((pred - off).abs().mean()), gate_alpha=float(saved),
            object_residual_norm=on_trace.get('object_residual_norm'), object_query_norm=on_trace.get('object_query_norm')))
    handle.remove()


def main():
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'pilot_started.json').exists(), 'Refusing accidental duplicate launch'
    t0 = time.time()
    hard_end = t0 + 9 * 3600
    cutoff = hard_end - 105 * 60
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)
    seed()
    train = json.loads((PRIOR / 'train90_episode_ids.json').read_text())
    dev = json.loads((PRIOR / 'dev15_episode_ids.json').read_text())
    write('pilot_started.json', dict(t0=t0, hard_end=hard_end, training_cutoff=cutoff, lease='user confirms no time limit', seed=1))
    # Gate file binds the supplied collection invocation and explicit future interface.
    unit = json.loads((OUT / 'unit_contract.json').read_text())
    assert unit['status'] in ('VERIFIED', 'FRESH_TRAIN_LIMITED'), 'UNIT_GATE_BLOCKED'
    stats = statistics(train)
    prototypes(train)
    cfg = config()
    cfg.save_pretrained(OUT / 'effective_model_config')
    write('effective_training_config.json', dict(seed=1, batch_size=4, num_workers=4, optimizer='AdamW',
        lr=1e-5, weight_decay=1e-4, grad_clip=10, scheduler=None, use_amp=False,
        chunk=100, smoke_steps=100, max_pilot_steps=80000, hard_end=hard_end,
        training_cutoff=cutoff, normalization='Train-only MEAN_STD; ImageNet visual stats',
        cudnn_benchmark=False, cudnn_deterministic=True, deterministic_algorithms=False,
        recipe_source='historical B3_config/B3_train_config and train_t3_joint_state_spatial.sh',
        augmentation='none', sample_resume='epoch/cursor deterministic permutation; no random image transforms'))
    dataset = Mapped(train)
    pre, post = make_act_pre_post_processors(cfg, stats)
    policy = ACTPolicy(cfg).cuda()
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-5, weight_decay=1e-4)
    raw = next(iter(loader(dataset, 0)))
    batch = pre(raw)
    assert batch['observation.state'].shape == (4, 8) and batch['action'].shape == (4, 100, 6)
    for k in ('action', 'observation.state', 'observation.images.top', 'observation.images.gripper'):
        assert torch.isfinite(batch[k]).all()
    state_mean, state_std = stats['observation.state']['mean'].cuda(), stats['observation.state']['std'].cuda()
    torch.testing.assert_close(batch['observation.state'], (raw['observation.state'].cuda() - state_mean) / state_std)
    torch.testing.assert_close(batch['observation.state'] * state_std + state_mean, raw['observation.state'].cuda(), atol=1e-4, rtol=1e-5)
    assert not any(p.requires_grad for p in policy.model.object_conditioning.dino.parameters())
    write('production_gate.json', dict(schema='PASS', normalization='one standard ACT preprocessor', unit=unit['status'],
                                     state_shape=[4,8], action_shape=[4,100,6], frozen_dino=True, prototype_train_only=True))
    epoch, cursor, step = 0, 0, 0
    smoke_start = time.time()
    for raw in loader(dataset, 0):
        policy.train()
        batch = pre(raw)
        loss, metrics = policy(batch)
        assert torch.isfinite(loss), 'NONFINITE_LOSS'
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10)
        assert torch.isfinite(grad), 'NONFINITE_GRAD'
        optimizer.step()
        step += 1
        row('smoke_log.csv', dict(step=step, loss=float(loss.detach()), grad_norm=float(grad), elapsed_s=time.time()-smoke_start))
        if step == 100:
            break
    path = checkpoint(policy, optimizer, 100, 0, 100, cfg, 'smoke')
    policy.eval()
    with torch.inference_mode():
        before = policy.predict_action_chunk(batch).cpu()
    saved = torch.load(path, weights_only=False, map_location='cpu')
    reloaded = ACTPolicy(cfg).cuda()
    reloaded.load_state_dict(saved['model'])
    restored_optimizer = torch.optim.AdamW(reloaded.get_optim_params(), lr=1e-5, weight_decay=1e-4)
    restored_optimizer.load_state_dict(saved['optimizer'])
    with torch.inference_mode():
        after = reloaded.predict_action_chunk(batch).cpu()
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    write('smoke_restore_validation.json', dict(step=100, same_input_max_abs_diff=float((before-after).abs().max()), restored_optimizer=True))
    del policy, optimizer, saved, reloaded, restored_optimizer
    torch.cuda.empty_cache()
    # Separate fresh formal trajectory, same seed/config; smoke is not counted.
    seed()
    policy = ACTPolicy(cfg).cuda()
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-5, weight_decay=1e-4)
    step, epoch, cursor = 0, 0, 0
    last_save = time.time()
    checkpoints = []
    while step < 80000 and time.time() < cutoff:
        for raw in loader(dataset, epoch, cursor):
            policy.train()
            batch = pre(raw)
            start = time.time()
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = policy(batch)
            assert torch.isfinite(loss), 'NONFINITE_LOSS'
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10)
            assert torch.isfinite(grad), 'NONFINITE_GRAD'
            optimizer.step()
            torch.cuda.synchronize()
            step += 1
            cursor += 1
            row('training_metrics.csv', dict(step=step, epoch=epoch, cursor=cursor, loss=float(loss.detach()), grad_norm=float(grad),
                compute_s=time.time()-start, elapsed_s=time.time()-t0, peak_memory_bytes=torch.cuda.max_memory_allocated()))
            if step in (10000,20000,40000,60000,80000) or time.time()-last_save >= 1800 or time.time() >= cutoff:
                checkpoints.append((step, checkpoint(policy, optimizer, step, epoch, cursor, cfg, 'pilot')))
                last_save = time.time()
            if step >= 80000 or time.time() >= cutoff:
                break
            if step == 500:
                write('throughput_500.json', dict(step=step, elapsed_since_t0=time.time()-t0,
                    compute_only_s_last_step=time.time()-start, peak_memory_bytes=torch.cuda.max_memory_allocated(),
                    note='Use training_metrics.csv wall-time differences for end-to-end sec/step, not compute-only timing'))
        else:
            epoch += 1
            cursor = 0
            continue
        break
    if not checkpoints or checkpoints[-1][0] != step:
        checkpoints.append((step, checkpoint(policy, optimizer, step, epoch, cursor, cfg, 'pilot')))
    choices = list(dict(checkpoints[-3:]).items())
    for saved_step, path in choices:
        saved = torch.load(path, weights_only=False, map_location='cpu')
        policy.load_state_dict(saved['model'])
        del saved
        evaluate(policy, pre, stats, dev, saved_step)
    write('execution_complete.json', dict(outcome='PILOT_READY_FOR_OFFLINE_SELECTION' if step>=80000 else 'PILOT_NEEDS_CONTINUATION',
                                         effective_steps=step, total_s=time.time()-t0, final_access=False))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'failure.log').write_text(traceback.format_exc())
        raise
