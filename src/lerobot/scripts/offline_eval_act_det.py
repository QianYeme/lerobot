#!/usr/bin/env python

# Copyright 2026 QianYeme. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Offline evaluation for trained ACT / ACTDet checkpoints on a local LeRobot dataset.

Computes the training-equivalent metrics on held-out episodes (no robot required):

  - action L1    : mean |action_hat - action| over the chunk, padded positions masked
  - detection loss: focal cls + reg + centerness components (act_det with use_detection)
  - mask loss     : L1 between pred_mask and SAM2 NPZ ground-truth masks
                    (act_det with use_mask_guidance)

The policy runs in train mode (under torch.no_grad()) so that the detection and
mask losses, which are gated by `training` in ACTDetModel.forward, are computed
exactly like during training. Batches go through the same ACT pre-processor
pipeline as in `lerobot_train.py` (normalization with dataset stats), and
episode_index/frame_index pass through unchanged so annotation lookups work.
Because train mode keeps dropout and the online top-camera augmentation active,
the numbers carry a small random noise — run twice and average if you need
tighter estimates.

Example:
```shell
python src/lerobot/scripts/offline_eval_act_det.py \
    --checkpoint outputs/train/2026-08-08/12-34-49_act_det/checkpointsE9/last/pretrained_model \
    --dataset.repo_id formal1_B \
    --dataset.root /home/lyj/lerobot/数据集/formal1_B \
    --episodes 63-89
```

The dataset is auto-detected from the checkpoint's `annotation_dir` for act_det
checkpoints; for plain `act` checkpoints (e.g. E1) pass `--dataset.repo_id` and
`--dataset.root` explicitly.
"""

import argparse
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act_det.modeling_act_det import ACTDetPolicy
from lerobot.policies.factory import make_pre_post_processors

POLICY_CLASSES = {"act": ACTPolicy, "act_det": ACTDetPolicy}

METRIC_KEYS = [
    "l1_loss",
    "kld_loss",
    "det_cls_loss",
    "det_reg_loss",
    "det_ctr_loss",
    "mask_loss",
]


def parse_episodes(spec: str) -> list[int]:
    """Parse an episode spec like '63-89' or '0,3,5-7' into a list of indices."""
    episodes = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            episodes.extend(range(int(lo), int(hi) + 1))
        else:
            episodes.append(int(part))
    return episodes


def infer_dataset_from_checkpoint(checkpoint: str | Path) -> tuple[str, str] | None:
    """Extract (repo_id, root) from the checkpoint's annotation_dir, if present."""
    config_path = Path(checkpoint) / "config.json"
    if not config_path.exists():
        return None
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    annotation_dir = config.get("annotation_dir")
    if not annotation_dir:
        return None
    # annotation_dir looks like ".../数据集/<repo_id>/annotations"
    m = re.search(r"数据集[\\/]([^\\/]+)", annotation_dir)
    if not m:
        return None
    # root must be the dataset folder itself (contains meta/data/videos),
    # i.e. the parent of the annotations directory.
    root = str(Path(annotation_dir).parent)
    return m.group(1), root


def collate(batch: list[dict]) -> dict:
    """Stack tensor fields, drop non-tensor fields (e.g. the 'task' string)."""
    out = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            out[key] = torch.stack([item[key] for item in batch])
    return out

def evaluate(checkpoint: Path, dataset: LeRobotDataset, batch_size: int,
             num_workers: int, annotation_dir: str | None = None) -> dict:
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    if cfg.type not in POLICY_CLASSES:
        raise ValueError(f"Unsupported policy type {cfg.type!r} in {checkpoint}")

    # Checkpoints trained on a remote machine (e.g. AutoDL) bake an absolute
    # `annotation_dir`/`mask_dir` into config.json that doesn't exist locally.
    # Let the caller override both so detection/mask losses can be computed on
    # a local copy of the dataset.
    if annotation_dir is not None:
        cfg.annotation_dir = annotation_dir
        cfg.mask_dir = f"{annotation_dir}/masks"

    logging.info("Loading policy %s from %s", cfg.type, checkpoint)
    policy = POLICY_CLASSES[cfg.type].from_pretrained(checkpoint, config=cfg)
    policy.train()  # required so detection/mask losses are computed

    # Load the SAVED pre-processor from the checkpoint so normalization matches
    # training exactly. Training runs with `use_imagenet_stats=True`, which overrides
    # image mean/std with ImageNet stats; rebuilding from `dataset.meta.stats` would
    # use the raw camera stats (std ~0.01) and amplify the images ~100x, so the
    # action L1 would be systematically ~10x too high.
    preprocessor, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(checkpoint))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=False,
    )

    accum = defaultdict(float)
    total_frames = 0
    start = time.perf_counter()

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            # The ACT pre-processor round-trips the batch through an EnvTransition,
            # which keeps episode_index/index/task_index but DROPS frame_index. The
            # ACTDet detection/mask losses need frame_index to look up per-frame
            # annotations, so save it before preprocessing and restore it afterwards.
            frame_index = batch.get("frame_index")
            batch = preprocessor(batch)
            if frame_index is not None:
                batch["frame_index"] = frame_index
            _, loss_dict = policy.forward(batch)
            bs = batch["observation.state"].shape[0]
            total_frames += bs
            for key in METRIC_KEYS:
                if key in loss_dict and loss_dict[key] is not None:
                    accum[key] += float(loss_dict[key]) * bs
            if step % 50 == 0:
                logging.info("Step %d (%d frames), %ds elapsed", step, total_frames,
                             time.perf_counter() - start)

    logging.info("Evaluated %d frames in %.1fs", total_frames, time.perf_counter() - start)
    return {key: accum[key] / total_frames for key in METRIC_KEYS if key in accum}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the pretrained_model directory")
    parser.add_argument("--dataset.repo_id", default=None,
                        help="Dataset id (e.g. formal_A). Auto-detected for act_det checkpoints.")
    parser.add_argument("--dataset.root", default=None,
                        help="Dataset folder itself, containing meta/data/videos "
                             "(e.g. .../数据集/formal1_B)")
    parser.add_argument("--episodes", default="63-89",
                        help="Episode spec to evaluate on, e.g. '63-89' (default: last 27 episodes)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dataset.video_backend", default="pyav",
                        help="Video decode backend (default 'pyav', matching training). "
                             "torchcodec needs system FFmpeg libs and fails on some machines.")
    parser.add_argument("--annotation-dir", default=None,
                        help="Override the checkpoint's annotation_dir (and derived mask_dir) "
                             "for local evaluation, e.g. /home/lyj/lerobot/数据集/formal1_B/annotations")
    parser.add_argument("--output", default=None,
                        help="JSON path for the results (default: <checkpoint>/offline_eval_results.json)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    checkpoint = Path(args.checkpoint)
    repo_id = getattr(args, "dataset.repo_id", None)
    root = getattr(args, "dataset.root", None)
    if repo_id is None or root is None:
        inferred = infer_dataset_from_checkpoint(checkpoint)
        if inferred is None and repo_id is None:
            raise SystemExit("Cannot infer the dataset from the checkpoint. "
                             "Pass --dataset.repo_id and --dataset.root explicitly.")
        if inferred is not None:
            repo_id, root = repo_id or inferred[0], root or inferred[1]
    logging.info("Dataset: %s (root=%s)", repo_id, root)

    episodes = parse_episodes(args.episodes)
    logging.info("Episodes: %d-%d (%d episodes)", episodes[0], episodes[-1], len(episodes))

    ds_meta = LeRobotDatasetMetadata(repo_id, root=root)
    chunk_size = json.load(open(checkpoint / "config.json", encoding="utf-8"))["chunk_size"]
    delta_timestamps = {"action": [i / ds_meta.fps for i in range(chunk_size)]}
    dataset = LeRobotDataset(
        repo_id, root=root, episodes=episodes, delta_timestamps=delta_timestamps,
        video_backend=getattr(args, "dataset.video_backend", "pyav"),
    )

    metrics = evaluate(checkpoint, dataset, args.batch_size, args.num_workers,
                       annotation_dir=getattr(args, "annotation_dir", None))

    print("\n===== Offline evaluation results =====")
    print(f"checkpoint : {checkpoint}")
    print(f"dataset    : {repo_id}, episodes {episodes[0]}-{episodes[-1]}")
    for key in METRIC_KEYS:
        if key in metrics:
            print(f"  {key:<15} {metrics[key]:.6f}")

    output = Path(args.output) if args.output else checkpoint / "offline_eval_results.json"
    output.write_text(json.dumps({
        "checkpoint": str(checkpoint),
        "dataset": repo_id,
        "episodes": [episodes[0], episodes[-1]],
        "metrics": metrics,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
