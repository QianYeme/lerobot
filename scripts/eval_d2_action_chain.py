"""Frozen Dev teacher-forced chunks and hardware-free 2x2 replay for D2."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.offline_eval_act_det import collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    started = time.monotonic()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(args.split).read_text())
    episodes = [e['episode_index'] for e in split['episodes'] if e['split'] == 'dev']
    assert len(episodes) == 12
    torch.manual_seed(0)
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.device = 'cuda'
    policy = ACTPolicy.from_pretrained(args.checkpoint, config=cfg).to('cuda').eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=args.checkpoint)
    ds = LeRobotDataset('formal4', root=args.root, episodes=episodes, video_backend='pyav')
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate)
    result = {}
    for batch_index, raw in enumerate(loader):
        if time.monotonic() - started >= 4 * 3600:
            raise RuntimeError('D2 GPU budget reached')
        ids = raw['episode_index'].numpy().reshape(-1)
        frames = raw['frame_index'].numpy().reshape(-1)
        states = raw['observation.state'].numpy()
        targets = raw['action'].numpy()
        batch = pre(raw)
        batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        # Inference must not receive action targets (VAE inference uses zero latent).
        batch.pop('action', None)
        with torch.no_grad():
            chunks = policy.predict_action_chunk(batch)
            normalized = chunks.cpu().numpy()
            physical = torch.stack([post(chunks[:, k]) for k in range(chunks.shape[1])], dim=1).cpu().numpy()
        for i, ep in enumerate(ids):
            result.setdefault(int(ep), []).append((int(frames[i]), states[i], targets[i], physical[i], normalized[i]))
        if batch_index % 100 == 0:
            print('inference', batch_index, 'batches', 'elapsed_s', round(time.monotonic()-started, 1), flush=True)
    metrics = {}
    for ep in episodes:
        rows = sorted(result[ep], key=lambda x: x[0])
        frames = np.array([r[0] for r in rows])
        state = np.stack([r[1] for r in rows])
        target = np.stack([r[2] for r in rows])
        chunks = np.stack([r[3] for r in rows])
        normal = np.stack([r[4] for r in rows])
        np.savez_compressed(out / f'episode_{ep:03d}.npz', frames=frames, state=state,
                            target=target, chunks=chunks, normalized_chunks=normal)
        replay = {}
        for coeff in (None, 0.01):
            ens = ACTTemporalEnsembler(coeff, cfg.chunk_size) if coeff is not None else None
            # Current contract freezes n_action_steps=1 for off mode.
            if coeff is None and cfg.n_action_steps != 1:
                raise ValueError('Off-mode action queue needs explicit reconstruction')
            preds = []
            for chunk in chunks:
                preds.append(ens.update(torch.from_numpy(chunk)[None])[0].numpy()
                             if ens else chunk[0])
            preds = np.stack(preds)
            for limit in (5, 2):
                # Equivalent per-joint clamp; input recorded state, not simulated rollout.
                sent = state[:, :6] + np.clip(preds - state[:, :6], -limit, limit)
                replay[f'{coeff}_{limit}'] = {
                    'per_joint_clip_rate': (np.abs(preds-state[:, :6]) > limit).mean(0).tolist(),
                    'per_joint_processing_delta': np.abs(sent-chunks[:, 0]).mean(0).tolist(),
                    'per_joint_mae': np.abs(sent-target).mean(0).tolist()}
        metrics[str(ep)] = {'frames': len(rows), 'phase': 'unknown',
                           'h0_per_joint_mae': np.abs(chunks[:, 0]-target).mean(0).tolist(),
                           'replay': replay}
        print('episode', ep, len(rows), flush=True)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in Path(args.checkpoint).iterdir() if p.is_file()}
    (out/'metrics.json').write_text(json.dumps({'episodes': metrics, 'checkpoint_hashes': hashes,
        'teacher_forced': True, 'phase_labels': 'UNKNOWN_PENDING_VISUAL_AUDIT',
        'full_chunk_ground_truth_evaluated': False, 'gpu_seconds': time.monotonic()-started,
        'final_test': False, 'training': False, 'real_robot': False}, indent=2))


if __name__ == '__main__':
    main()
