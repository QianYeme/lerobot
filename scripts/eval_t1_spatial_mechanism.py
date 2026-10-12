#!/usr/bin/env python
"""T1 fixed-Dev evaluation and S0-S4 explicit-spatial ablations."""
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


def move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def l1_parts(pred: torch.Tensor, target: torch.Tensor, pad: torch.Tensor) -> dict[str, float]:
    if target.ndim == 2 and pred.ndim == 3:
        pred = pred[:, 0]
    if target.ndim == 2 and pad.ndim == 2:
        pad = pad[:, 0]
    err = (pred - target).abs() * (~pad).unsqueeze(-1)
    h0 = err
    arm = err[..., :-1]
    gripper = err[..., -1:]
    return {"all": float(err.mean()), "h0_all": float(h0.mean()),
            "arm": float(arm.mean()), "h0_arm": float(arm.mean()),
            "gripper": float(gripper.mean()), "h0_gripper": float(gripper.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--episodes", default=",".join(map(str, DEV_EPISODES)))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()
    episodes = parse_episodes(args.episodes)
    if episodes != DEV_EPISODES:
        raise ValueError(f"T1 is frozen to Dev episodes {DEV_EPISODES}, got {episodes}")
    checkpoint = Path(args.checkpoint)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    policy = ACTPolicy.from_pretrained(checkpoint, config=cfg); policy.eval()
    device = next(policy.parameters()).device
    pre, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(checkpoint))
    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=episodes, video_backend="pyav")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    collate_fn=collate, drop_last=False)
    oc = getattr(policy.model, "object_conditioning", None)
    top_idx = getattr(policy.model, "object_conditioning_top_camera_index", None)
    is_p2 = bool(getattr(cfg, "explicit_spatial_conditioning", False))
    sums = {k: {name: 0.0 for name in ("all", "h0_all", "arm", "h0_arm", "gripper", "h0_gripper")}
            for k in (["S0_normal", "S1_zero", "S2_shuffle", "S3_x_only", "S4_y_only"] if is_p2 else ["S0_normal"])}
    count = 0
    intervention_rows: list[dict] = []
    generator = torch.Generator(device=device).manual_seed(17)

    with torch.no_grad():
        for raw_batch in dl:
            # The ACT preprocessor may drop padding metadata because it is
            # primarily an inference input transform. Keep the dataset's
            # action padding mask for teacher-forced Dev metrics.
            raw_action_is_pad = raw_batch.get("action_is_pad")
            batch = pre(raw_batch)
            if "action_is_pad" not in batch and raw_action_is_pad is not None:
                batch["action_is_pad"] = raw_action_is_pad
            if "action_is_pad" not in batch:
                # Fixed-length Dev samples have no padding field in this
                # dataset export; all requested action targets are valid.
                batch["action_is_pad"] = torch.zeros(
                    batch[ACTION].shape[:-1], dtype=torch.bool, device=batch[ACTION].device
                )
            batch = move(batch, device)
            if cfg.image_features:
                batch[OBS_IMAGES] = [batch[k] for k in cfg.image_features]
            target = batch[ACTION]
            pad = batch["action_is_pad"]
            if oc is not None:
                original_extract = oc.extract_raw_object_tokens
                top_image = batch[OBS_IMAGES][top_idx]
                base_raw, similarity = original_extract(top_image)
                permutation = torch.randperm(base_raw.shape[0], generator=generator, device=device)
                original_spatial = getattr(oc, "global_spatial_features")
            else:
                base_raw = similarity = permutation = original_extract = original_spatial = None

            def run(mode: str) -> torch.Tensor:
                if oc is None or not is_p2 or mode == "S0_normal":
                    return policy.model(batch)[0]
                donor = base_raw[permutation, :, :]
                donor_spatial = donor[:, 0, 384:387]
                base_spatial = base_raw[:, 0, 384:387]
                if mode == "S1_zero":
                    chosen = torch.zeros_like(base_spatial)
                elif mode == "S2_shuffle":
                    chosen = donor_spatial
                elif mode == "S3_x_only":
                    chosen = base_spatial.clone(); chosen[:, 0] = donor_spatial[:, 0]
                elif mode == "S4_y_only":
                    chosen = base_spatial.clone(); chosen[:, 1] = donor_spatial[:, 1]
                else:
                    raise ValueError(mode)
                oc.extract_raw_object_tokens = lambda _top: (base_raw, similarity)
                oc.global_spatial_features = lambda _raw: chosen
                try:
                    return policy.model(batch)[0]
                finally:
                    oc.extract_raw_object_tokens = original_extract
                    oc.global_spatial_features = original_spatial

            predictions = {mode: run(mode) for mode in sums}
            for mode, pred in predictions.items():
                parts = l1_parts(pred, target, pad)
                for key, value in parts.items(): sums[mode][key] += value * len(target)
            if is_p2:
                normal_h0 = predictions["S0_normal"][:, 0]
                for mode in ("S1_zero", "S2_shuffle", "S3_x_only", "S4_y_only"):
                    delta = (predictions[mode][:, 0] - normal_h0).detach().cpu().numpy()
                    base_spatial = base_raw[:, 0, 384:387]
                    donor_spatial = base_raw[permutation, 0, 384:387]
                    spatial_delta = donor_spatial - base_spatial
                    if mode == "S1_zero":
                        spatial_delta = -base_spatial
                    elif mode == "S3_x_only":
                        spatial_delta = torch.stack((spatial_delta[:, 0], torch.zeros_like(spatial_delta[:, 0]), spatial_delta[:, 2]), dim=1)
                    elif mode == "S4_y_only":
                        spatial_delta = torch.stack((torch.zeros_like(spatial_delta[:, 0]), spatial_delta[:, 1], spatial_delta[:, 2]), dim=1)
                    intervention_rows.append({"mode": mode, "h0_delta": delta.tolist(),
                                              "spatial_delta": spatial_delta.detach().cpu().tolist(),
                                              "mean_abs_arm": float(np.mean(np.abs(delta[:, :-1]))),
                                              "mean_abs_gripper": float(np.mean(np.abs(delta[:, -1])))})
            count += len(target)

    for mode in sums:
        for key in sums[mode]: sums[mode][key] /= max(count, 1)
    result = {"checkpoint": str(checkpoint), "model": "P2" if is_p2 else "B1",
              "episodes": episodes, "frames": count, "device": str(device), "metrics": sums,
              "final_test_used": False, "real_robot_used": False}
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "spatial_ablations.json").write_text(json.dumps(intervention_rows, indent=2), encoding="utf-8")
    (out / "validation.json").write_text(json.dumps({
        "dev_only": True, "episodes_frozen": episodes == DEV_EPISODES,
        "checkpoint": str(checkpoint), "p2_explicit_path": is_p2,
        "dino_frozen": bool(getattr(cfg, "object_conditioning_dino_frozen", True)),
        "tau": float(getattr(cfg, "object_conditioning_temperature", -1.0)),
    }, indent=2), encoding="utf-8")
    print(json.dumps({"model": result["model"], "frames": count, "metrics": sums}, indent=2))


if __name__ == "__main__":
    main()
