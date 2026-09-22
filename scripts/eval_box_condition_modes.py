#!/usr/bin/env python

"""Evaluate token, state-fusion, and action-residual box conditioning modes."""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import torch

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.offline_eval_act_det import collate, parse_episodes
from smoke_c0_explicit_box import build_c0_smoke_policy, first_frame_indices


def same_direction(values: list[float], reference: list[float]) -> bool:
    return all((b - a) * (rb - ra) > 0 for (a, b), (ra, rb) in zip(pairwise(values), pairwise(reference)))


def condition_module(policy):
    mode = policy.config.box_condition_mode
    if mode == "token":
        return policy.model.box_condition_input_proj
    if mode == "state":
        return policy.model.box_condition_state_proj
    if mode == "action_residual":
        return policy.model.box_condition_action_residual
    raise ValueError(f"unsupported box condition mode: {mode}")


def run_with_condition(policy, batch, transform=None) -> torch.Tensor:
    handle = None
    if transform is not None:
        mode = policy.config.box_condition_mode

        def replace(_module, inputs):
            value = inputs[0]
            if mode == "action_residual":
                return (torch.cat((value[:, :-6], transform(value[:, -6:])), dim=-1),)
            return (transform(value),)

        handle = condition_module(policy).register_forward_pre_hook(replace)
    try:
        with torch.inference_mode():
            return policy.predict_action_chunk(batch).detach()
    finally:
        if handle is not None:
            handle.remove()


def max_pan_effect(reference: torch.Tensor, candidate: torch.Tensor, pan_index: int) -> float:
    return float((reference[:, 0, pan_index] - candidate[:, 0, pan_index]).abs().max())


def run(args: argparse.Namespace) -> dict:
    episodes = parse_episodes(args.episodes)
    policy, load_report = build_c0_smoke_policy(args.checkpoint, args.device, args.score_threshold)
    if load_report["load_mode"] != "native_c0":
        raise ValueError("mode evaluation requires a native explicit-box checkpoint")
    config = policy.config

    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        episodes=episodes,
        delta_timestamps={"action": [i / metadata.fps for i in range(config.chunk_size)]},
        video_backend=args.dataset_video_backend,
    )
    raw_batch = collate([dataset[index] for index in first_frame_indices(dataset, episodes)])
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    batch = preprocessor(raw_batch)
    center = args.fixed_state_index

    fixed_batch = dict(batch)
    fixed_batch["observation.state"] = batch["observation.state"][center : center + 1].expand_as(
        batch["observation.state"]
    )
    for camera in config.image_features:
        if camera != config.box_condition_camera:
            fixed_batch[camera] = batch[camera][center : center + 1].expand_as(batch[camera])

    baseline = run_with_condition(policy, fixed_batch)
    condition = policy.model.get_box_condition().detach().clone()
    baseline_residual, baseline_alpha = policy.model.get_box_action_residual()
    zero = run_with_condition(policy, fixed_batch, lambda value: torch.zeros_like(value))
    reversed_box = run_with_condition(policy, fixed_batch, lambda value: value.flip(0))

    coordinate_only_batch = dict(fixed_batch)
    coordinate_only_batch[config.box_condition_camera] = batch[config.box_condition_camera][
        center : center + 1
    ].expand_as(batch[config.box_condition_camera])
    coordinate_actions = {
        "normal": run_with_condition(policy, coordinate_only_batch, lambda _value: condition),
        "zero": run_with_condition(policy, coordinate_only_batch, lambda value: torch.zeros_like(value)),
        "reversed": run_with_condition(policy, coordinate_only_batch, lambda _value: condition.flip(0)),
    }

    gt_pan = batch["action"][:, 0, args.pan_index].detach().cpu().tolist()
    baseline_pan = baseline[:, 0, args.pan_index].cpu().tolist()
    coordinate_pan = {
        name: actions[:, 0, args.pan_index].cpu().tolist()
        for name, actions in coordinate_actions.items()
    }
    coordinate_span = max(coordinate_pan["normal"]) - min(coordinate_pan["normal"])
    zero_effect = max_pan_effect(baseline, zero, args.pan_index)
    reverse_effect = max_pan_effect(baseline, reversed_box, args.pan_index)
    visible = condition[:, 5].eq(1).all().item()
    cx_ordered = all(left < right for left, right in pairwise(condition[:, 0].cpu().tolist()))
    direction_pass = same_direction(coordinate_pan["normal"], gt_pan)
    span_pass = coordinate_span >= args.min_pan_span
    effect_pass = max(zero_effect, reverse_effect) >= args.min_box_effect

    residual_report = None
    if baseline_residual is not None:
        residual_report = {
            "alpha": float(baseline_alpha.detach()),
            "abs_mean": float(baseline_residual.abs().mean()),
            "abs_max": float(baseline_residual.abs().max()),
        }

    result = {
        "status": "passed" if visible and cx_ordered and direction_pass and span_pass and effect_pass else "failed",
        "checkpoint": str(args.checkpoint),
        "mode": config.box_condition_mode,
        "episodes_left_to_right": episodes,
        "protocol": {
            "fixed_state_episode": episodes[center],
            "fixed_non_box_camera_episode": episodes[center],
            "coordinate_only_images_episode": episodes[center],
            "action_step": 0,
            "pan_index": args.pan_index,
            "min_pan_span": args.min_pan_span,
            "min_box_effect": args.min_box_effect,
        },
        "gt_h0_pan": gt_pan,
        "predicted_condition": condition.cpu().tolist(),
        "baseline_h0_pan": baseline_pan,
        "zero_h0_pan": zero[:, 0, args.pan_index].cpu().tolist(),
        "reversed_h0_pan": reversed_box[:, 0, args.pan_index].cpu().tolist(),
        "coordinate_only_h0_pan": coordinate_pan,
        "residual": residual_report,
        "gates": {
            "all_boxes_visible": bool(visible),
            "cx_strictly_increasing": cx_ordered,
            "coordinate_only_direction_matches_gt": direction_pass,
            "coordinate_only_span": coordinate_span,
            "coordinate_only_span_pass": span_pass,
            "zero_h0_pan_effect": zero_effect,
            "reverse_h0_pan_effect": reverse_effect,
            "box_effect_pass": effect_pass,
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", required=True)
    parser.add_argument("--dataset.root", dest="dataset_root", required=True)
    parser.add_argument("--episodes", default="33,10,6")
    parser.add_argument("--fixed-state-index", type=int, default=1)
    parser.add_argument("--pan-index", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--min-pan-span", type=float, default=0.05)
    parser.add_argument("--min-box-effect", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset.video_backend", dest="dataset_video_backend", default="pyav")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0 <= args.fixed_state_index < len(parse_episodes(args.episodes)):
        raise ValueError("fixed-state-index is outside the episode list")

    result = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
