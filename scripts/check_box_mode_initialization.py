#!/usr/bin/env python

"""Verify exact shared initialization for C0/C1/C2 full-size policies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.policies.act_det.configuration_act_det import ACTDetConfig
from lerobot.policies.act_det.modeling_act_det import ACTDetPolicy


def canonical_shared_state(policy: ACTDetPolicy) -> dict[str, torch.Tensor]:
    shared = {}
    for key, value in policy.state_dict().items():
        if key.startswith("model.box_condition_"):
            continue
        tensor = value.detach().cpu().clone()
        if key == "model.encoder_1d_feature_pos_embed.weight":
            tensor = tensor[:2]
        shared[key] = tensor
    return shared


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def make_config(metadata: LeRobotDatasetMetadata, mode: str) -> ACTDetConfig:
    features = dataset_to_policy_features(metadata.features)
    output_features = {key: value for key, value in features.items() if value.type is FeatureType.ACTION}
    input_features = {key: value for key, value in features.items() if key not in output_features}
    return ACTDetConfig(
        device="cpu",
        input_features=input_features,
        output_features=output_features,
        chunk_size=100,
        n_action_steps=1,
        gripper_loss_weight=3.0,
        use_detection=True,
        use_mask_guidance=False,
        fcos_feature_inject=False,
        mask_feature_inject=False,
        use_explicit_box_condition=True,
        box_condition_mode=mode,
        box_condition_camera="observation.images.top",
        box_condition_score_threshold=0.25,
        box_condition_dropout=0.0,
        box_condition_noise_std=0.0,
        box_action_residual_alpha=0.05,
        aug_enable=False,
        det_weight=1.0,
        det_cameras={
            "observation.images.top": {"enable": True},
            "observation.images.gripper": {"enable": False},
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", required=True)
    parser.add_argument("--dataset.root", dest="dataset_root", required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    reference = None
    reference_mode = None
    modes = {}
    mismatches = []
    for mode in ("token", "state", "action_residual"):
        torch.manual_seed(args.seed)
        policy = ACTDetPolicy(make_config(metadata, mode))
        shared = canonical_shared_state(policy)
        modes[mode] = {
            "shared_sha256": state_hash(shared),
            "shared_tensor_count": len(shared),
            "total_parameters": sum(parameter.numel() for parameter in policy.parameters()),
        }
        if reference is None:
            reference = shared
            reference_mode = mode
        else:
            if set(shared) != set(reference):
                mismatches.append(f"{mode}: shared key set differs from {reference_mode}")
            for key in sorted(set(shared) & set(reference)):
                if not torch.equal(shared[key], reference[key]):
                    mismatches.append(f"{mode}: {key}")
        del policy, shared

    result = {
        "status": "passed" if not mismatches else "failed",
        "seed": args.seed,
        "dataset": {"repo_id": args.dataset_repo_id, "root": args.dataset_root},
        "modes": modes,
        "mismatches": mismatches,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
