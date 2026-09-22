#!/usr/bin/env python

"""Run a non-training C0 interface smoke on real LeRobot frames.

For a legacy pure-DET checkpoint, compatible weights are copied into a C0
instance while the new coordinate MLP/token position stay randomly initialized.
That path checks only schema, decoding and execution; it is not a warm-started
C0 model and its actions are not evaluation evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
from itertools import pairwise
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act_det.modeling_act_det import ACTDetPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.offline_eval_act_det import collate, parse_episodes


POSITION_KEY = "model.encoder_1d_feature_pos_embed.weight"


def build_c0_smoke_policy(
    checkpoint: str | Path,
    device: str,
    score_threshold: float,
) -> tuple[ACTDetPolicy, dict]:
    """Load native C0, or make an explicitly labelled legacy DET smoke copy."""
    config = PreTrainedConfig.from_pretrained(checkpoint)
    if config.type != "act_det" or not config.use_detection:
        raise ValueError("checkpoint must be an ACTDet policy with detection enabled")
    if config.fcos_feature_inject or config.use_mask_guidance or config.mask_feature_inject:
        raise ValueError("C0 control must be a pure DET checkpoint, not an injection or MASK variant")
    config.device = device

    if config.use_explicit_box_condition:
        policy = ACTDetPolicy.from_pretrained(checkpoint, config=config, strict=True)
        policy.config.box_condition_score_threshold = score_threshold
        return policy.eval(), {"load_mode": "native_c0", "randomly_initialized": []}

    source = ACTDetPolicy.from_pretrained(checkpoint, config=config, strict=True)
    c0_config = copy.deepcopy(config)
    c0_config.use_explicit_box_condition = True
    c0_config.box_condition_mode = "token"
    c0_config.box_condition_score_threshold = score_threshold
    c0_config.box_condition_dropout = 0.0
    c0_config.box_condition_noise_std = 0.0
    policy = ACTDetPolicy(c0_config).to(device)

    source_state = source.state_dict()
    target_state = policy.state_dict()
    copied = []
    partial_random = []
    for key, value in source_state.items():
        if key == POSITION_KEY:
            if value.shape[1:] != target_state[key].shape[1:]:
                raise ValueError(f"incompatible position embedding: {value.shape} -> {target_state[key].shape}")
            if value.shape[0] >= target_state[key].shape[0]:
                raise ValueError("legacy checkpoint does not leave one new position for the C0 token")
            target_state[key][: value.shape[0]].copy_(value)
            copied.append(key)
            partial_random.append(f"{key}[{value.shape[0]}:]")
        elif key in target_state and value.shape == target_state[key].shape:
            target_state[key].copy_(value)
            copied.append(key)
        else:
            raise ValueError(f"unexpected incompatible legacy weight: {key} {tuple(value.shape)}")
    policy.load_state_dict(target_state, strict=True)
    policy.eval()
    random_keys = sorted(set(target_state) - set(copied)) + partial_random
    return policy, {
        "load_mode": "legacy_det_interface_only",
        "randomly_initialized": random_keys,
        "warning": "Actions from this copied model are not C0 performance evidence.",
    }


def first_frame_indices(dataset: LeRobotDataset, episodes: list[int]) -> list[int]:
    wanted = set(episodes)
    found: dict[int, int] = {}
    episode_column = dataset.hf_dataset["episode_index"]
    frame_column = dataset.hf_dataset["frame_index"]
    for index, (episode, frame) in enumerate(zip(episode_column, frame_column, strict=True)):
        episode = int(episode)
        if episode in wanted and int(frame) == 0:
            found[episode] = index
            if len(found) == len(wanted):
                break
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"missing frame 0 for episodes: {sorted(missing)}")
    return [found[episode] for episode in episodes]


def run(args: argparse.Namespace) -> dict:
    episodes = parse_episodes(args.episodes)
    policy, load_report = build_c0_smoke_policy(
        args.checkpoint, args.device, args.score_threshold
    )
    config = policy.config

    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    state_feature = metadata.features["observation.state"]
    state_names = state_feature.get("names") or []
    state_dim = state_feature["shape"][0]
    if state_dim != args.expected_state_dim:
        raise ValueError(f"state dim is {state_dim}, expected {args.expected_state_dim}")
    if args.expected_state_dim == 8 and "master_gripper.pos" in state_names:
        raise ValueError("8D NOMASTER smoke still contains master_gripper.pos")
    policy_state_dim = config.input_features["observation.state"].shape[0]
    if policy_state_dim != state_dim:
        raise ValueError(f"checkpoint state dim {policy_state_dim} != dataset state dim {state_dim}")

    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        episodes=episodes,
        video_backend=args.dataset_video_backend,
    )
    samples = [dataset[index] for index in first_frame_indices(dataset, episodes)]
    raw_batch = collate(samples)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    batch = preprocessor(raw_batch)

    with torch.inference_mode():
        actions = policy.predict_action_chunk(batch)
    condition = policy.model.get_box_condition()
    if condition is None or condition.shape != (len(episodes), 6):
        raise RuntimeError(f"invalid box condition shape: {None if condition is None else condition.shape}")
    if not torch.isfinite(actions).all() or not torch.isfinite(condition).all():
        raise RuntimeError("non-finite action or box condition")

    camera = config.box_condition_camera
    height, width = batch[camera].shape[-2:]
    rows = []
    for episode, values in zip(episodes, condition.cpu().tolist(), strict=True):
        cx, cy, box_w, box_h, confidence, visible = values
        rows.append(
            {
                "episode": episode,
                "frame": 0,
                "normalized": {
                    "cx": cx,
                    "cy": cy,
                    "w": box_w,
                    "h": box_h,
                    "confidence": confidence,
                    "visible": visible,
                },
                "pixels": {
                    "cx": (cx + 1) * width / 2 if visible else None,
                    "cy": (cy + 1) * height / 2 if visible else None,
                    "w": box_w * width if visible else None,
                    "h": box_h * height if visible else None,
                },
            }
        )

    if args.expect_cx_increasing:
        if any(row["normalized"]["visible"] != 1 for row in rows):
            raise RuntimeError("cx ordering cannot pass because at least one prediction is not visible")
        centers = [row["normalized"]["cx"] for row in rows]
        if not all(left < right for left, right in pairwise(centers)):
            raise RuntimeError(f"expected strictly increasing cx in episode order, got {centers}")

    return {
        "status": "passed",
        "scope": "P2 interface/coordinate smoke only",
        "checkpoint": str(args.checkpoint),
        "dataset": {"repo_id": args.dataset_repo_id, "root": str(args.dataset_root)},
        "episodes": episodes,
        "state_dim": state_dim,
        "state_names": state_names,
        "camera": camera,
        "processed_image_size": [height, width],
        "action_shape": list(actions.shape),
        "load": load_report,
        "conditions": rows,
        "cx_increasing_checked": args.expect_cx_increasing,
        "limitations": [
            "No optimizer step or training was run.",
            "Finite actions do not establish action quality.",
            "Legacy DET copy mode cannot establish C0 performance.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", required=True)
    parser.add_argument("--dataset.root", dest="dataset_root", required=True)
    parser.add_argument("--episodes", default="0,2,4")
    parser.add_argument("--expected-state-dim", type=int, default=8)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument(
        "--expect-cx-increasing",
        action="store_true",
        help="Require episodes to be supplied in left-to-right order and decoded cx to increase.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset.video_backend", dest="dataset_video_backend", default="pyav")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = run(args)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
