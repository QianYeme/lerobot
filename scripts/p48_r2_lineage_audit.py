"""P4.8-r2 read-only lineage gate and real-frame normalization arithmetic audit.

No model is instantiated, no prototype is constructed, and no optimizer runs.
"""
import hashlib
import json
from pathlib import Path


ROOT = Path('/root/autodl-tmp/lerobot-main')
PRIOR = ROOT / 'experiments/p4_public_benchmark/p4_8_libero_pilot/index1/result'
OUT = ROOT / 'experiments/p4_public_benchmark/p4_8_libero_pilot/index3/result'
DATA = Path('/root/autodl-tmp/p48_data/lerobot_libero')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    import sys
    sys.path.insert(0, str(PRIOR / 'workspace/lerobot/src'))
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    selected = json.loads((PRIOR / 'selected_task.json').read_text())
    ids = sorted(selected['episode_ids'])
    permutation = np.random.default_rng(1).permutation(ids).tolist()
    split = dict(dataset='lerobot/libero', revision='a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4',
                 task_index=24, simulator_suite='libero_object', simulator_task_id=0,
                 seed=1, algorithm='numpy Generator(PCG64) permutation(sorted episode IDs); first35 Train, last9 Dev',
                 numpy_version=np.__version__, train_episode_ids=sorted(permutation[:35]),
                 dev_episode_ids=sorted(permutation[35:]), status='FROZEN_BEFORE_MODEL_TRAINING',
                 no_prior_formal_split=True, source_sha256=digest(PRIOR / 'selected_task.json'),
                 caveat='G1 already inspected three episodes for data engineering; Dev is not blind test data.')
    assert len(ids) == 44 and len(set(ids)) == 44
    assert not set(split['train_episode_ids']) & set(split['dev_episode_ids'])
    assert sorted(split['train_episode_ids'] + split['dev_episode_ids']) == ids
    path = OUT / 'dataset_split_manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == split, 'Never replace an existing split'
    else:
        write(path.name, split)

    old = ROOT / 'experiments/p4_final_p1_training/p4_1_unattended_p1_pilot/index3/result'
    train_file = ROOT / 'experiments/p4_final_p1_training/p4_0_1_training_readiness_unblock/index1/result/train90_episode_ids.json'
    train = json.loads(train_file.read_text())
    provenance = json.loads((old / 'prototype_provenance.json').read_text())
    # Only Train membership is read. No Final15 file, image, state or action is accessed.
    if isinstance(train, dict):
        raise RuntimeError('Unexpected historical Train manifest schema; inspect before changing audit')
    assert all(p['episode'] in train for p in provenance)
    write('historical_train_membership.json', dict(status='PASS', train_manifest_sha256=digest(train_file),
          prototype_provenance_sha256=digest(old / 'prototype_provenance.json'),
          selected_episode_ids=[p['episode'] for p in provenance], all_in_train90=True,
          final15_accessed=False))

    available = json.loads((PRIOR / 'video_sanity_selection.json').read_text())['episode_ids']
    ep = next(ep for ep in available if ep in split['train_episode_ids'])
    dataset = LeRobotDataset('lerobot/libero', root=DATA, episodes=[ep],
                             revision=split['revision'], video_backend='torchcodec')
    raw = dataset[0]
    assert raw['observation.state'].shape == (8,) and raw['action'].shape == (7,)
    image = raw['observation.images.image'].unsqueeze(0)
    # Official LIBERO image convention: flip both spatial axes once.
    image = image.flip((-2, -1))
    mean = image.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
    std = image.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
    fixed = (F.interpolate(image, (224, 224), mode='bilinear', align_corners=False) - mean) / std
    legacy = (F.interpolate((image - mean) / std, (224, 224), mode='bilinear', align_corners=False) - mean) / std
    prototype = (F.interpolate(image, (224, 224), mode='bilinear', align_corners=False) - mean) / std
    def stats(x):
        assert torch.isfinite(x).all()
        return dict(shape=list(x.shape), dtype=str(x.dtype), minimum=float(x.min()), maximum=float(x.max()),
                    channel_mean=x.mean((0, 2, 3)).tolist(), channel_std=x.std((0, 2, 3)).tolist(),
                    tensor_sha256=hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest())
    assert torch.equal(fixed, prototype) and not torch.equal(fixed, legacy)
    write('legacy_fixed_preprocess_probe.json', dict(status='ARITHMETIC_REAL_TRAIN_FRAME_PASS',
          scope='Reference normalization arithmetic only; NOT a P1 forward hook or effective implementation validation',
          episode=ep, frame=0, split_sha256=digest(path), image_flip_axes=['height', 'width'],
          raw=stats(image), fixed_online=stats(fixed), legacy_online=stats(legacy),
          prototype_side=stats(prototype), fixed_prototype_max_difference=float((fixed-prototype).abs().max()),
          legacy_fixed_max_difference=float((legacy-fixed).abs().max()),
          p1_instantiated=False, dino_forward_executed=False, train_updates=0))
    meta = pd.read_parquet(DATA / 'meta/episodes/chunk-000/file-000.parquet')
    lengths = {str(ep): int(meta.loc[meta.episode_index == ep, 'length'].iloc[0]) for ep in ids}
    write('split_validation.json', dict(status='PASS', train_episodes=35, dev_episodes=9,
          train_frames=sum(lengths[str(ep)] for ep in split['train_episode_ids']),
          dev_frames=sum(lengths[str(ep)] for ep in split['dev_episode_ids']), episode_lengths=lengths,
          overlap=[], selected_total_frames=sum(lengths.values()), metadata_sha256=digest(DATA / 'meta/episodes/chunk-000/file-000.parquet')))
    print('R2_LINEAGE_AND_REFERENCE_PREPROCESS_AUDIT_DONE; G2 REMAINS BLOCKED', flush=True)


if __name__ == '__main__':
    main()
