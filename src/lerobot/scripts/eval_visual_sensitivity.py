#!/usr/bin/env python

"""Measure whether an ACT/ACTDet checkpoint actually uses the top-camera image.

For each held-out batch, this script compares action chunks predicted from:
1. the original observation;
2. the same state with top-camera images cyclically swapped across the batch;
3. the same state with a neutral (zero after normalization) top-camera image.

A visually conditioned policy should change its actions and should normally incur
higher action error when the image is swapped or blanked. Results are reported in
normalized action space, matching the model's training loss.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.offline_eval_act_det import (
    POLICY_CLASSES,
    collate,
    infer_dataset_from_checkpoint,
    parse_episodes,
)


def _abs_stats(prediction: torch.Tensor, reference: torch.Tensor, is_pad: torch.Tensor) -> tuple[float, int]:
    valid = (~is_pad).unsqueeze(-1).expand_as(prediction)
    return torch.abs(prediction - reference)[valid].sum().item(), int(valid.sum().item())


def evaluate(
    checkpoint: Path,
    dataset: LeRobotDataset,
    camera_key: str,
    batch_size: int,
    num_workers: int,
    max_batches: int | None,
    action_steps: int | None,
    seed: int,
) -> dict:
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2 so images can be swapped")

    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    if cfg.type not in POLICY_CLASSES:
        raise ValueError(f"Unsupported policy type {cfg.type!r} in {checkpoint}")
    if camera_key not in cfg.image_features:
        raise ValueError(f"{camera_key!r} is not an input camera; available: {list(cfg.image_features)}")

    policy = POLICY_CLASSES[cfg.type].from_pretrained(checkpoint, config=cfg)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(checkpoint))

    generator = torch.Generator().manual_seed(seed)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=True,
    )

    totals = defaultdict(float)
    counts = defaultdict(int)
    batches = 0
    with torch.no_grad():
        for batch in dataloader:
            if max_batches is not None and batches >= max_batches:
                break
            batch = preprocessor(batch)
            steps = action_steps or batch["action"].shape[1]
            steps = min(steps, batch["action"].shape[1])
            target = batch["action"][:, :steps]
            is_pad = batch["action_is_pad"][:, :steps]

            original = policy.predict_action_chunk(batch)[:, :steps]

            swapped_batch = dict(batch)
            swapped_batch[camera_key] = torch.roll(batch[camera_key], shifts=1, dims=0)
            swapped = policy.predict_action_chunk(swapped_batch)[:, :steps]

            blank_batch = dict(batch)
            blank_batch[camera_key] = torch.zeros_like(batch[camera_key])
            blank = policy.predict_action_chunk(blank_batch)[:, :steps]

            comparisons = {
                "original_action_l1": (original, target),
                "swapped_action_l1": (swapped, target),
                "blank_action_l1": (blank, target),
                "swap_action_delta_l1": (swapped, original),
                "blank_action_delta_l1": (blank, original),
            }
            for name, (prediction, reference) in comparisons.items():
                value_sum, value_count = _abs_stats(prediction, reference, is_pad)
                totals[name] += value_sum
                counts[name] += value_count
            batches += 1

    if batches == 0:
        raise RuntimeError("No complete batches were available for evaluation")

    metrics = {name: totals[name] / counts[name] for name in totals}
    metrics["swap_error_increase"] = metrics["swapped_action_l1"] - metrics["original_action_l1"]
    metrics["blank_error_increase"] = metrics["blank_action_l1"] - metrics["original_action_l1"]
    return {"batches": batches, "frames": batches * batch_size, "action_steps": steps, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset.repo_id", default=None)
    parser.add_argument("--dataset.root", default=None)
    parser.add_argument("--episodes", default="63-89")
    parser.add_argument("--camera-key", default="observation.images.top")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset.video_backend", default="pyav")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    checkpoint = Path(args.checkpoint)
    repo_id = getattr(args, "dataset.repo_id", None)
    root = getattr(args, "dataset.root", None)
    if repo_id is None or root is None:
        inferred = infer_dataset_from_checkpoint(checkpoint)
        if inferred is None:
            raise SystemExit("Cannot infer dataset; pass --dataset.repo_id and --dataset.root")
        repo_id, root = repo_id or inferred[0], root or inferred[1]

    episodes = parse_episodes(args.episodes)
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    chunk_size = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))["chunk_size"]
    dataset = LeRobotDataset(
        repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps={"action": [i / metadata.fps for i in range(chunk_size)]},
        video_backend=getattr(args, "dataset.video_backend", "pyav"),
    )
    result = evaluate(
        checkpoint,
        dataset,
        args.camera_key,
        args.batch_size,
        args.num_workers,
        args.max_batches,
        args.action_steps,
        args.seed,
    )
    result.update({"checkpoint": str(checkpoint), "dataset": repo_id, "episodes": episodes})

    print("\n===== Visual sensitivity results =====")
    for key, value in result["metrics"].items():
        print(f"  {key:<25} {value:.6f}")
    output = Path(args.output) if args.output else checkpoint / "visual_sensitivity_results.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
