"""Two-update real-data smoke for optional ACTDet phase supervision; not training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act_det.configuration_act_det import ACTDetConfig
from lerobot.policies.act_det.modeling_act_det import ACTDetPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.offline_eval_act_det import collate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--phase-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    torch.manual_seed(1000)
    metadata = LeRobotDatasetMetadata("formal3/kind_merged", root=args.dataset_root)
    features = dataset_to_policy_features(metadata.features)
    output_features = {key: value for key, value in features.items() if value.type is FeatureType.ACTION}
    input_features = {key: value for key, value in features.items() if key not in output_features}
    annotation_dir = args.dataset_root / "annotations"
    config = ACTDetConfig(
        device=args.device,
        input_features=input_features,
        output_features=output_features,
        pretrained_backbone_weights=None,
        chunk_size=100,
        n_action_steps=1,
        use_detection=True,
        annotation_dir=str(annotation_dir) if annotation_dir.is_dir() else None,
        use_mask_guidance=False,
        fcos_feature_inject=False,
        aug_enable=False,
        use_phase_aux=True,
        phase_labels=str(args.phase_labels),
        phase_weight=0.1,
    )
    policy = ACTDetPolicy(config).to(args.device).train()
    preprocessor, _ = make_pre_post_processors(config, dataset_stats=metadata.stats)
    dataset = LeRobotDataset(
        "formal3/kind_merged",
        root=args.dataset_root,
        episodes=[0],
        delta_timestamps={"action": [index / metadata.fps for index in range(config.chunk_size)]},
        video_backend="pyav",
    )
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-5)
    steps = []
    for frame in (0, 200):
        raw = collate([dataset[frame]])
        identity = {key: raw[key].clone() for key in ("episode_index", "frame_index")}
        batch = preprocessor(raw)
        batch.update({key: value.to(args.device) for key, value in identity.items()})
        optimizer.zero_grad(set_to_none=True)
        loss, logs = policy(batch)
        if not torch.isfinite(loss) or logs["phase_valid_frames"] != 1:
            raise AssertionError(f"Invalid phase smoke loss: {logs}")
        if annotation_dir.is_dir() and (logs["det_reg_loss"] <= 0 or logs["det_ctr_loss"] <= 0):
            raise AssertionError(f"Attached detection annotations produced no positive regression target: {logs}")
        phase_loss = policy.model._phase_loss["phase_loss"]
        phase_loss.backward(retain_graph=True)
        phase_grad = policy.model.phase_head.weight.grad
        backbone_grads = [p.grad for p in policy.model.backbone.parameters() if p.grad is not None]
        if phase_grad is None or not torch.isfinite(phase_grad).all() or phase_grad.abs().sum() == 0:
            raise AssertionError("Phase head did not receive finite nonzero gradients")
        if not backbone_grads or not all(torch.isfinite(gradient).all() for gradient in backbone_grads):
            raise AssertionError("Backbone did not receive finite phase-only gradients")
        phase_gradient_norm = float(phase_grad.square().sum().sqrt())
        backbone_phase_gradient_norm = sum(float(gradient.square().sum()) for gradient in backbone_grads) ** 0.5
        if backbone_phase_gradient_norm == 0:
            raise AssertionError("Backbone phase-only gradient is zero")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        steps.append({
            "frame": frame,
            "loss": float(loss.detach()),
            "phase_gradient_norm": phase_gradient_norm,
            "backbone_phase_gradient_norm": backbone_phase_gradient_norm,
            "components": logs,
        })

    policy.eval()
    deployment = {key: value for key, value in batch.items() if key not in {"action", "action_is_pad", "episode_index", "frame_index"}}
    with torch.inference_mode():
        actions = policy.predict_action_chunk(deployment)
    if actions.shape != (1, config.chunk_size, 6) or not torch.isfinite(actions).all():
        raise AssertionError("Deployment action interface changed or produced invalid values")

    limitations = []
    if not annotation_dir.is_dir():
        limitations.append(
            "formal3 detection annotations are not yet attached, so this smoke does not validate FCOS target supervision"
        )
    if input_features["observation.state"].shape[0] != 8:
        limitations.insert(
            0,
            "formal3 still contains master_gripper.pos; prepare an immutable NOMASTER derivative before training",
        )
    report = {
        "status": "passed",
        "scope": "two optimizer updates on real formal3 frames; not a training or quality result",
        "frames": [0, 200],
        "phase_ids": [0, 1],
        "state_dim": input_features["observation.state"].shape[0],
        "action_dim": output_features["action"].shape[0],
        "action_shape": list(actions.shape),
        "deployment_without_labels": True,
        "detection_annotations_attached": annotation_dir.is_dir(),
        "steps": steps,
        "limitations": limitations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
