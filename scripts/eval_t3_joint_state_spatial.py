#!/usr/bin/env python
"""T3 fixed-Dev joint state-spatial counterfactual evaluation."""
from __future__ import annotations

import argparse
import json
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


DEV_EPISODES = [3, 5, 12, 15, 21, 28, 33, 39, 41, 47, 53, 58]
MODES = ("J0_normal", "J1_spatial_zero", "J2_xy_shuffle", "J3_x_transplant", "J4_y_transplant", "J5_full_spatial_transplant")


def move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def h0(pred: torch.Tensor) -> torch.Tensor:
    return pred[:, 0] if pred.ndim == 3 else pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", required=True); ap.add_argument("--episodes", default=','.join(map(str, DEV_EPISODES)))
    ap.add_argument("--output-dir", required=True); ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()
    episodes = parse_episodes(args.episodes)
    if episodes != DEV_EPISODES:
        raise ValueError(f"T3 is frozen to Dev episodes {DEV_EPISODES}, got {episodes}")
    ckpt = Path(args.checkpoint); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    policy = ACTPolicy.from_pretrained(ckpt, config=cfg); policy.eval()
    device = next(policy.parameters()).device
    pre, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(ckpt))
    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=episodes, video_backend="pyav")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate, drop_last=False)
    is_joint = bool(getattr(cfg, "joint_state_spatial_conditioning", False))
    oc = getattr(policy.model, "object_conditioning", None)
    top_idx = getattr(policy.model, "object_conditioning_top_camera_index", None)
    modes = MODES if is_joint else ("J0_normal",)
    sums = {m: {k: 0.0 for k in ("all", "h0_all", "arm", "gripper")} for m in modes}
    dim_sum = {m: None for m in modes}; strata = {m: {"left": [0.0, 0], "middle": [0.0, 0], "right": [0.0, 0]} for m in modes}
    intervention_rows = []; count = 0; generator = torch.Generator(device=device).manual_seed(17)
    with torch.no_grad():
        for raw in dl:
            raw_pad = raw.get("action_is_pad")
            batch = pre(raw)
            if "action_is_pad" not in batch and raw_pad is not None: batch["action_is_pad"] = raw_pad
            batch = move(batch, device)
            if "action_is_pad" not in batch: batch["action_is_pad"] = torch.zeros(batch[ACTION].shape[:-1], dtype=torch.bool, device=device)
            if cfg.image_features: batch[OBS_IMAGES] = [batch[k] for k in cfg.image_features]
            target = h0(batch[ACTION]); pad = batch["action_is_pad"]; pad = pad[:, 0] if pad.ndim == 2 else pad
            base_raw = similarity = permutation = None
            if is_joint:
                top = batch[OBS_IMAGES][top_idx]; base_raw, similarity = oc.extract_raw_object_tokens(top)
                permutation = torch.randperm(base_raw.shape[0], generator=generator, device=device)
            original_extract = oc.extract_raw_object_tokens if oc is not None else None
            original_spatial = oc.global_spatial_features if oc is not None else None

            def run(mode: str) -> torch.Tensor:
                if not is_joint or mode == "J0_normal": return h0(policy.model(batch)[0])
                donor = base_raw[permutation]
                base_sp = base_raw[:, 0, 384:387]; donor_sp = donor[:, 0, 384:387]
                chosen = base_sp.clone()
                if mode == "J1_spatial_zero": chosen.zero_()
                elif mode == "J2_xy_shuffle" or mode == "J5_full_spatial_transplant": chosen = donor_sp
                elif mode == "J3_x_transplant": chosen[:, 0] = donor_sp[:, 0]
                elif mode == "J4_y_transplant": chosen[:, 1] = donor_sp[:, 1]
                else: raise ValueError(mode)
                oc.extract_raw_object_tokens = lambda _top: (base_raw, similarity)
                oc.global_spatial_features = lambda _raw: chosen
                try: return h0(policy.model(batch)[0])
                finally:
                    oc.extract_raw_object_tokens = original_extract; oc.global_spatial_features = original_spatial

            preds = {m: run(m) for m in modes}
            x = base_raw[:, 0, 384] if is_joint else None
            for mode, pred in preds.items():
                err = (pred - target).abs() * (~pad).unsqueeze(-1)
                sums[mode]["all"] += float(err.mean()) * len(target); sums[mode]["h0_all"] += float(err.mean()) * len(target)
                sums[mode]["arm"] += float(err[:, :-1].mean()) * len(target); sums[mode]["gripper"] += float(err[:, -1].mean()) * len(target)
                d = err.mean(dim=0).detach().cpu().numpy(); dim_sum[mode] = d if dim_sum[mode] is None else dim_sum[mode] + d * len(target)
                if is_joint:
                    for name, mask in (("left", x < -0.1), ("middle", (x >= -0.1) & (x <= 0.1)), ("right", x > 0.1)):
                        if mask.any(): strata[mode][name][0] += float(err[mask].mean()) * int(mask.sum()); strata[mode][name][1] += int(mask.sum())
            if is_joint:
                normal = preds["J0_normal"]
                for mode in MODES[1:]:
                    delta = (preds[mode] - normal).cpu().numpy()
                    intervention_rows.append({"mode": mode, "h0_delta": delta.tolist(), "mean_abs_arm": float(np.mean(np.abs(delta[:, :-1]))), "mean_abs_gripper": float(np.mean(np.abs(delta[:, -1]))), "spatial_delta": (base_raw[permutation, 0, 384:387] - base_raw[:, 0, 384:387]).cpu().tolist()})
            count += len(target)
    for m in modes:
        for k in sums[m]: sums[m][k] /= max(count, 1)
        sums[m]["per_action_dim"] = (dim_sum[m] / max(count, 1)).tolist()
        sums[m]["position_stratified"] = {k: (v[0] / max(v[1], 1)) for k, v in strata[m].items()}
    model = "P4" if is_joint else "B3"
    result = {"experiment": "T3_JOINT_STATE_SPATIAL_CONDITIONING", "checkpoint": str(ckpt), "model": model, "episodes": episodes, "frames": count, "device": str(device), "joint_path": is_joint, "metrics": sums, "final_test_used": False, "real_robot_used": False}
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "joint_counterfactual.json").write_text(json.dumps(intervention_rows, indent=2), encoding="utf-8")
    (out / "validation.json").write_text(json.dumps({"dev_only": True, "episodes_frozen": episodes == DEV_EPISODES, "checkpoint": str(ckpt), "joint_path": is_joint, "dino_frozen": bool(getattr(cfg, "object_conditioning_dino_frozen", True)), "tau": float(getattr(cfg, "object_conditioning_temperature", -1.0)), "final_test_used": False, "real_robot_used": False}, indent=2), encoding="utf-8")
    print(json.dumps({"model": model, "frames": count, "metrics": sums}, indent=2))


if __name__ == "__main__": main()
