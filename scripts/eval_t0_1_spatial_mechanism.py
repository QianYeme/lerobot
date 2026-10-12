#!/usr/bin/env python
"""T0.1 OC-ACT spatial mechanism diagnostics on the frozen Dev split."""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.offline_eval_act_det import collate, parse_episodes
from lerobot.utils.constants import ACTION, OBS_IMAGES

MODES = ["D0_normal", "D1_zero_all", "D2_shuffle_whole", "D3_xy_zero",
         "D4_xy_shuffle", "D5_lr_swap", "D6_ud_swap", "D7_local_permutation",
         "D8_global_only", "D9_local_only"]


def corr(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    out = []
    for j in range(y.shape[1]):
        if np.std(x) < 1e-9 or np.std(y[:, j]) < 1e-9:
            out.append(0.0)
        else:
            out.append(float(np.corrcoef(x, y[:, j])[0, 1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    ckpt = Path(args.checkpoint)
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    policy = ACTPolicy.from_pretrained(ckpt, config=cfg)
    policy.eval()
    device = next(policy.parameters()).device
    pre, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(ckpt))
    episodes = parse_episodes(args.episodes)
    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=episodes, video_backend="pyav")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4,
                    collate_fn=collate, drop_last=False)
    oc = policy.model.object_conditioning
    top_idx = policy.model.object_conditioning_top_camera_index
    rng = np.random.default_rng(17)
    sums = {m: {"all": 0.0, "arm": 0.0, "gripper": 0.0, "h0_all": 0.0,
                "h0_arm": 0.0, "h0_gripper": 0.0, "n": 0} for m in MODES}
    coords, target_h0, pred_h0 = [], [], []

    def forward_with_tokens(model_batch, tokens):
        original = oc.forward
        try:
            def injected(act_tokens, top_image, object_tokens=None):
                kv = tokens.transpose(0, 1)
                response, _ = oc.cross_attention(act_tokens, kv, kv, need_weights=False)
                return act_tokens + oc.gate * response
            oc.forward = injected
            return policy.model(model_batch)[0]
        finally:
            oc.forward = original

    def build_tokens(raw, mode):
        if mode == "D1_zero_all":
            return oc.adapt_object_tokens(raw).zero_()
        if mode == "D2_shuffle_whole":
            return oc.adapt_object_tokens(raw.flip(0))
        if mode == "D3_xy_zero":
            r = raw.clone(); r[..., 384:386] = 0.0
            return oc.adapt_object_tokens(r)
        if mode == "D4_xy_shuffle":
            r = raw.clone(); r[..., 384:386] = r.flip(0)[..., 384:386]
            return oc.adapt_object_tokens(r)
        if mode == "D5_lr_swap":
            r = raw.clone(); r[:, [1, 2]] = r[:, [2, 1]]; r[:, [3, 4]] = r[:, [4, 3]]
            return oc.adapt_object_tokens(r)
        elif mode == "D6_ud_swap":
            r = raw.clone(); r[:, [1, 3]] = r[:, [3, 1]]; r[:, [2, 4]] = r[:, [4, 2]]
            return oc.adapt_object_tokens(r)
        elif mode == "D7_local_permutation":
            # np.random.default_rng(17).permutation([1,2,3,4]) -> [1,3,2,4].
            return oc.adapt_object_tokens(raw[:, [0, 1, 3, 2, 4]])
        tokens = oc.adapt_object_tokens(raw)
        if mode == "D8_global_only":
            tokens = tokens.clone(); tokens[:, 1:] = 0.0
        elif mode == "D9_local_only":
            tokens = tokens.clone(); tokens[:, 0] = 0.0
        return tokens

    with torch.no_grad():
        for batch in dl:
            batch = pre(batch)
            model_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if cfg.image_features:
                model_batch[OBS_IMAGES] = [model_batch[k] for k in cfg.image_features]
            top_image = model_batch[OBS_IMAGES][top_idx]
            raw, _ = oc.extract_raw_object_tokens(top_image)
            normal = policy.model(model_batch)[0]
            target = model_batch[ACTION]
            normal_h0 = normal[:, 0] if normal.ndim == 3 else normal
            target_h0_batch = target[:, 0] if target.ndim == 3 else target
            coords.extend(raw[:, 0, 384:386].detach().cpu().tolist())
            target_h0.extend(target_h0_batch.detach().cpu().tolist())
            pred_h0.extend(normal_h0.detach().cpu().tolist())
            mode_preds = {"D0_normal": normal}
            for mode in MODES[1:]:
                mode_preds[mode] = forward_with_tokens(model_batch, build_tokens(raw, mode))
            for mode, pred in mode_preds.items():
                delta = (pred - normal).abs() if mode != "D0_normal" else torch.zeros_like(normal)
                h0 = delta[:, 0] if delta.ndim == 3 else delta
                n = h0.shape[0]
                sums[mode]["all"] += float(delta.abs().mean()) * n
                sums[mode]["arm"] += float(delta[..., :-1].abs().mean()) * n
                sums[mode]["gripper"] += float(delta[..., -1].abs().mean()) * n
                sums[mode]["h0_all"] += float(h0.mean()) * n
                sums[mode]["h0_arm"] += float(h0[..., :-1].mean()) * n
                sums[mode]["h0_gripper"] += float(h0[..., -1].mean()) * n
                sums[mode]["n"] += n

    for mode in MODES:
        n = max(1, sums[mode]["n"])
        for k in ("all", "arm", "gripper", "h0_all", "h0_arm", "h0_gripper"):
            sums[mode][k] /= n
    coords, target_h0, pred_h0 = np.asarray(coords), np.asarray(target_h0), np.asarray(pred_h0)
    mapping = {
        "coordinate_source": "P1 raw global object token x/y, indices 384/385; no post-hoc dimension selection",
        "expert_action_h0_corr": {"x": corr(coords[:, 0], target_h0), "y": corr(coords[:, 1], target_h0)},
        "p1_predicted_action_h0_corr": {"x": corr(coords[:, 0], pred_h0), "y": corr(coords[:, 1], pred_h0)},
        "n_frames": int(len(coords)),
    }
    result = {
        "checkpoint": str(ckpt), "episodes": episodes, "seed": 17,
        "interventions": sums,
        "gate": float(oc.gate.detach().cpu()),
        "notes": "D5/D6 swap complete local tokens; D7 fixed permutation [global, LL, LR, UL, UR]; D0 is the normal reference.",
    }
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "intervention_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "action_position_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    (out / "transplant_pairs.json").write_text(json.dumps({
        "status": "not_run", "reason": "T0.1 required D0-D9 first; pair construction requires a separate matched-frame pass and was not used to select action dimensions."
    }, indent=2), encoding="utf-8")
    print(json.dumps({"interventions": result["interventions"], "mapping": mapping}, indent=2))


if __name__ == "__main__":
    main()
