#!/usr/bin/env python

"""Diagnose whether a trained C0 policy uses its explicit box token."""

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
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from smoke_c0_explicit_box import build_c0_smoke_policy, first_frame_indices


def tensor_norm_rows(tensor: torch.Tensor) -> list[float]:
    return tensor.norm(dim=-1).detach().cpu().tolist()


def same_direction(values: list[float], reference: list[float]) -> bool:
    return all((b - a) * (rb - ra) > 0 for (a, b), (ra, rb) in zip(pairwise(values), pairwise(reference)))


def action_delta(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    delta = (candidate - reference).abs().detach().cpu()
    return {
        "max_abs_all": float(delta.max()),
        "mean_abs_all": float(delta.mean()),
        "max_abs_h0": float(delta[:, 0].max()),
        "max_abs_h0_pan": float(delta[:, 0, 0].max()),
        "max_abs_pan_horizon": float(delta[:, :, 0].max()),
        "h0_pan": candidate[:, 0, 0].detach().cpu().tolist(),
        "h0_per_action_max": delta[:, 0].max(dim=0).values.tolist(),
    }


def run_with_box_hook(policy, batch, *, input_transform=None, output_transform=None):
    handles = []
    if input_transform is not None:
        handles.append(
            policy.model.box_condition_input_proj.register_forward_pre_hook(
                lambda _module, inputs: (input_transform(inputs[0]),)
            )
        )
    if output_transform is not None:
        handles.append(
            policy.model.box_condition_input_proj.register_forward_hook(
                lambda _module, _inputs, output: output_transform(output)
            )
        )
    try:
        with torch.inference_mode():
            return policy.predict_action_chunk(batch).detach()
    finally:
        for handle in handles:
            handle.remove()


def run(args: argparse.Namespace) -> dict:
    episodes = parse_episodes(args.episodes)
    policy, load_report = build_c0_smoke_policy(args.checkpoint, args.device, args.score_threshold)
    if load_report["load_mode"] != "native_c0":
        raise ValueError("diagnosis requires a trained native C0 checkpoint")

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
    fixed_batch = dict(batch)
    center = args.fixed_state_index
    fixed_batch[OBS_STATE] = batch[OBS_STATE][center : center + 1].expand_as(batch[OBS_STATE])
    for camera in config.image_features:
        if camera != config.box_condition_camera:
            fixed_batch[camera] = batch[camera][center : center + 1].expand_as(batch[camera])

    captured = {}

    def capture_encoder_input(_module, inputs):
        captured["encoder_input"] = inputs[0].detach().clone()

    def capture_encoder_output(_module, _inputs, output):
        captured["encoder_output"] = output.detach().clone()

    handles = [
        policy.model.encoder.register_forward_pre_hook(capture_encoder_input),
        policy.model.encoder.register_forward_hook(capture_encoder_output),
    ]
    try:
        baseline = run_with_box_hook(policy, fixed_batch)
        condition = policy.model.get_box_condition().detach().clone()
    finally:
        for handle in handles:
            handle.remove()

    variants = {
        "zero_raw_condition": run_with_box_hook(
            policy, fixed_batch, input_transform=lambda value: torch.zeros_like(value)
        ),
        "reverse_raw_condition": run_with_box_hook(
            policy, fixed_batch, input_transform=lambda value: value.flip(0)
        ),
        "zero_final_token": run_with_box_hook(
            policy, fixed_batch, output_transform=lambda value: torch.zeros_like(value)
        ),
        "reverse_final_token": run_with_box_hook(
            policy, fixed_batch, output_transform=lambda value: value.flip(0)
        ),
        "scale_final_token_10x": run_with_box_hook(
            policy, fixed_batch, output_transform=lambda value: value * 10
        ),
    }

    coordinate_only_batch = dict(fixed_batch)
    coordinate_only_batch[config.box_condition_camera] = batch[config.box_condition_camera][
        center : center + 1
    ].expand_as(batch[config.box_condition_camera])
    coordinate_actions = {
        "normal": run_with_box_hook(
            policy, coordinate_only_batch, input_transform=lambda _value: condition
        ),
        "reversed": run_with_box_hook(
            policy, coordinate_only_batch, input_transform=lambda _value: condition.flip(0)
        ),
        "zero_final_token": run_with_box_hook(
            policy,
            coordinate_only_batch,
            input_transform=lambda _value: condition,
            output_transform=lambda value: torch.zeros_like(value),
        ),
        "scale_final_token_10x": run_with_box_hook(
            policy,
            coordinate_only_batch,
            input_transform=lambda _value: condition,
            output_transform=lambda value: value * 10,
        ),
    }
    coordinate_pan = {
        name: value[:, 0, args.pan_index].detach().cpu().tolist()
        for name, value in coordinate_actions.items()
    }
    gt_pan = batch["action"][:, 0, args.pan_index].detach().cpu().tolist()

    base_1d_count = 1 + int(config.robot_state_feature is not None) + int(config.env_state_feature is not None)
    box_index = base_1d_count
    encoder_input = captured["encoder_input"]
    encoder_output = captured["encoder_output"]
    first_spatial_index = box_index + 1

    model_batch = dict(fixed_batch)
    model_batch[OBS_IMAGES] = [fixed_batch[key] for key in config.image_features]
    gradient_capture = {}

    def capture_projected_token(_module, _inputs, output):
        output.retain_grad()
        gradient_capture["projected_token"] = output

    def capture_encoder_tensor(_module, inputs):
        inputs[0].retain_grad()
        gradient_capture["encoder_input"] = inputs[0]

    policy.model.zero_grad(set_to_none=True)
    handles = [
        policy.model.box_condition_input_proj.register_forward_hook(capture_projected_token),
        policy.model.encoder.register_forward_pre_hook(capture_encoder_tensor),
    ]
    try:
        actions, _ = policy.model(model_batch, compute_aux_losses=False)
        actions[:, 0, args.pan_index].sum().backward()
    finally:
        for handle in handles:
            handle.remove()

    projected_grad = gradient_capture["projected_token"].grad
    encoder_grad = gradient_capture["encoder_input"].grad
    input_token_norms = encoder_input.norm(dim=-1).mean(dim=1)
    output_token_norms = encoder_output.norm(dim=-1).mean(dim=1)
    gradient_token_norms = encoder_grad.norm(dim=-1).mean(dim=1)

    deltas = {name: action_delta(baseline, value) for name, value in variants.items()}
    max_h0_pan_effect = max(
        deltas["zero_final_token"]["max_abs_h0_pan"],
        deltas["reverse_final_token"]["max_abs_h0_pan"],
    )
    result = {
        "status": "diagnosed",
        "checkpoint": str(args.checkpoint),
        "episodes_left_to_right": episodes,
        "protocol": {
            "fixed_state_episode": episodes[center],
            "fixed_non_box_camera_episode": episodes[center],
            "varied_image": config.box_condition_camera,
            "action_step": 0,
            "pan_index": args.pan_index,
        },
        "condition": condition.cpu().tolist(),
        "gt_h0_pan": gt_pan,
        "baseline_h0_pan": baseline[:, 0, args.pan_index].cpu().tolist(),
        "variant_deltas": deltas,
        "coordinate_only": {
            "description": "state and both images fixed to the center episode; only the injected condition varies",
            "h0_pan": coordinate_pan,
            "normal_span": max(coordinate_pan["normal"]) - min(coordinate_pan["normal"]),
            "normal_direction_matches_gt": same_direction(coordinate_pan["normal"], gt_pan),
            "scaled_10x_direction_matches_gt": same_direction(coordinate_pan["scale_final_token_10x"], gt_pan),
            "normal_vs_zero": action_delta(
                coordinate_actions["zero_final_token"], coordinate_actions["normal"]
            ),
            "normal_vs_reversed": action_delta(
                coordinate_actions["reversed"], coordinate_actions["normal"]
            ),
        },
        "token_norms": {
            "encoder_input_box_per_sample": tensor_norm_rows(encoder_input[box_index]),
            "encoder_input_state_mean": float(input_token_norms[1]) if config.robot_state_feature else None,
            "encoder_input_spatial_mean": float(input_token_norms[first_spatial_index:].mean()),
            "encoder_output_box_mean": float(output_token_norms[box_index]),
            "encoder_output_spatial_mean": float(output_token_norms[first_spatial_index:].mean()),
        },
        "h0_pan_gradient": {
            "projected_box_token_l2_per_sample": tensor_norm_rows(projected_grad),
            "encoder_input_box_mean_l2": float(gradient_token_norms[box_index]),
            "encoder_input_state_mean_l2": float(gradient_token_norms[1]) if config.robot_state_feature else None,
            "encoder_input_spatial_mean_l2": float(gradient_token_norms[first_spatial_index:].mean()),
            "encoder_input_spatial_max_l2": float(gradient_token_norms[first_spatial_index:].max()),
        },
        "diagnostic_flags": {
            "full_token_ablation_reaches_original_gate": deltas["zero_final_token"]["max_abs_h0_pan"] >= 0.01,
            "token_permutation_reaches_original_gate": deltas["reverse_final_token"]["max_abs_h0_pan"] >= 0.01,
            "any_h0_box_intervention_reaches_original_gate": max_h0_pan_effect >= 0.01,
            "effect_exists_outside_h0_pan": max(
                deltas["zero_final_token"]["max_abs_all"],
                deltas["reverse_final_token"]["max_abs_all"],
            ) >= 0.01,
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset.video_backend", dest="dataset_video_backend", default="pyav")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    episodes = parse_episodes(args.episodes)
    if not 0 <= args.fixed_state_index < len(episodes):
        raise ValueError("fixed-state-index is outside the episode list")
    result = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
