#!/usr/bin/env python

"""Evaluate C0 first-step pan sensitivity with only the top image varying."""

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


def strictly_same_direction(values: list[float], reference: list[float]) -> bool:
    return all((b - a) * (rb - ra) > 0 for (a, b), (ra, rb) in zip(pairwise(values), pairwise(reference)))


def run(args: argparse.Namespace) -> dict:
    episodes = parse_episodes(args.episodes)
    policy, load_report = build_c0_smoke_policy(args.checkpoint, args.device, args.score_threshold)
    if load_report["load_mode"] != "native_c0":
        raise ValueError("position sensitivity requires a trained native C0 checkpoint")
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
    gt_pan = batch["action"][:, 0, args.pan_index].detach().cpu().tolist()

    center = args.fixed_state_index
    fixed_batch = dict(batch)
    fixed_batch["observation.state"] = batch["observation.state"][center : center + 1].expand_as(
        batch["observation.state"]
    )
    for camera in config.image_features:
        if camera != config.box_condition_camera:
            fixed_batch[camera] = batch[camera][center : center + 1].expand_as(batch[camera])

    with torch.inference_mode():
        normal_actions = policy.predict_action_chunk(fixed_batch)
    condition = policy.model.get_box_condition().detach().cpu()

    handle = policy.model.box_condition_input_proj.register_forward_pre_hook(
        lambda _module, inputs: (torch.zeros_like(inputs[0]),)
    )
    try:
        with torch.inference_mode():
            zero_actions = policy.predict_action_chunk(fixed_batch)
    finally:
        handle.remove()

    normal_pan = normal_actions[:, 0, args.pan_index].detach().cpu().tolist()
    zero_pan = zero_actions[:, 0, args.pan_index].detach().cpu().tolist()
    box_effect = [normal - zero for normal, zero in zip(normal_pan, zero_pan, strict=True)]
    pan_span = max(normal_pan) - min(normal_pan)
    max_box_effect = max(abs(value) for value in box_effect)
    direction_pass = strictly_same_direction(normal_pan, gt_pan)
    span_pass = pan_span >= args.min_pan_span
    box_effect_pass = max_box_effect >= args.min_box_effect

    result = {
        "status": "passed" if direction_pass and span_pass and box_effect_pass else "failed",
        "checkpoint": str(args.checkpoint),
        "episodes_left_to_right": episodes,
        "protocol": {
            "fixed_state_episode": episodes[center],
            "fixed_non_top_camera_episode": episodes[center],
            "varied_input": config.box_condition_camera,
            "action_step": 0,
            "pan_index": args.pan_index,
            "min_pan_span": args.min_pan_span,
            "min_box_effect": args.min_box_effect,
        },
        "gt_h0_pan_normalized": gt_pan,
        "normal_h0_pan_normalized": normal_pan,
        "zero_box_h0_pan_normalized": zero_pan,
        "normal_minus_zero": box_effect,
        "predicted_cx_pixels": [float((row[0] + 1) * batch[config.box_condition_camera].shape[-1] / 2) for row in condition],
        "confidence": condition[:, 4].tolist(),
        "gates": {
            "direction_matches_demonstrations": direction_pass,
            "normal_pan_span": pan_span,
            "pan_span_pass": span_pass,
            "max_abs_box_effect": max_box_effect,
            "box_effect_pass": box_effect_pass,
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
