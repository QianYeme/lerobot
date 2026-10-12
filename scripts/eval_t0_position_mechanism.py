#!/usr/bin/env python
"""T0 Dev position and OC-ACT mechanism diagnostics (no Final Test)."""
import argparse, json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.offline_eval_act_det import collate, parse_episodes
from lerobot.utils.constants import ACTION, OBS_IMAGES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--repo-id', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--episodes', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--batch-size', type=int, default=4)
    args = ap.parse_args()
    ckpt = Path(args.checkpoint)
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    policy = ACTPolicy.from_pretrained(ckpt, config=cfg)
    policy.eval()
    device = next(policy.parameters()).device
    pre, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(ckpt))
    episodes = parse_episodes(args.episodes)
    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=episodes, video_backend='pyav')
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4,
                    collate_fn=collate, drop_last=False)
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    strata = {int(v['episode_index']): v.get('stratum', 'unknown') for v in manifest['episodes']}
    groups = defaultdict(lambda: {'n': 0, 'abs': 0.0, 'arm_abs': 0.0, 'gripper_abs': 0.0})
    mech = {'label': args.label, 'checkpoint': str(ckpt), 'n_batches': 0,
            'normal_zero_abs': [], 'normal_shuffle_abs': [], 'normal_spatial_swap_abs': [],
            'normal_l1': [], 'gate': None}
    oc = getattr(policy.model, 'object_conditioning', None)
    if oc is not None:
        mech['gate'] = float(oc.gate.detach().cpu())

    def run_mode(model_batch, mode='normal'):
        if oc is None or mode == 'normal':
            return policy.model(model_batch)[0]
        top_index = policy.model.object_conditioning_top_camera_index
        raw, _ = oc.extract_raw_object_tokens(model_batch[OBS_IMAGES][top_index])
        if mode == 'zero':
            tokens = oc.adapt_object_tokens(raw).zero_()
        else:
            if mode == 'shuffle':
                raw = raw.flip(0)
            elif mode == 'spatial_swap':
                raw = raw.clone(); raw[..., 384] = -raw[..., 384]
            tokens = oc.adapt_object_tokens(raw)
        original = oc.forward
        try:
            oc.forward = lambda act_tokens, top_image, object_tokens=None: act_tokens + oc.gate * oc.cross_attention(
                act_tokens, tokens.transpose(0, 1), tokens.transpose(0, 1), need_weights=False)[0]
            return policy.model(model_batch)[0]
        finally:
            oc.forward = original

    with torch.no_grad():
        for batch in dl:
            ep = batch['episode_index'].clone()
            batch = pre(batch)
            model_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if cfg.image_features:
                model_batch[OBS_IMAGES] = [model_batch[k] for k in cfg.image_features]
            pred = run_mode(model_batch)
            target = model_batch[ACTION]
            if pred.ndim == 3:
                pred0 = pred[:, 0]
                target0 = target[:, 0] if target.ndim == 3 else target
            else:
                pred0 = pred
                target0 = target[:, 0] if target.ndim == 3 else target
            err = (pred0 - target0).abs()
            for i, e in enumerate(ep.tolist()):
                g = groups[strata.get(int(e), 'unknown')]
                g['n'] += 1; g['abs'] += float(err[i].mean()); g['arm_abs'] += float(err[i, :-1].mean()); g['gripper_abs'] += float(err[i, -1])
            mech['normal_l1'].append(float(err.mean()))
            if oc is not None:
                for mode, key in [('zero','normal_zero_abs'), ('shuffle','normal_shuffle_abs'), ('spatial_swap','normal_spatial_swap_abs')]:
                    alt = run_mode(model_batch, mode)
                    mech[key].append(float((pred - alt).abs().mean()))
            mech['n_batches'] += 1
    for g in groups.values():
        n = max(g['n'], 1)
        for k in ('abs','arm_abs','gripper_abs'): g[k] /= n
    mech['normal_l1_mean'] = sum(mech['normal_l1']) / max(len(mech['normal_l1']), 1)
    for k in ('normal_zero_abs','normal_shuffle_abs','normal_spatial_swap_abs'):
        vals = mech[k]; mech[k + '_mean'] = sum(vals) / max(len(vals), 1)
    out = {'label': args.label, 'checkpoint': str(ckpt), 'episodes': episodes,
           'position_diagnostics': dict(groups), 'mechanism': mech}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__': main()
