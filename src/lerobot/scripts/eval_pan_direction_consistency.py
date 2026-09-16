#!/usr/bin/env python

"""Check whether shoulder_pan commands follow the cup's left-right position.

Complements eval_visual_sensitivity.py: that script shows the policy *responds*
to the top image; this one tests whether the response is *directionally
correct*. When the cup sits further right in the image, the demonstrations
imply a specific sign for the pan command; a policy that uses vision properly
should respond in that same direction.

Protocol (matches eval_visual_sensitivity.py so numbers are comparable):
- fixed validation episodes, batch_size=8, shuffled with a fixed seed,
  first `action_steps` steps of each predicted chunk, normalized action space;
- per sampled frame three pan values (mean over valid steps):
    gt_pan   : ground-truth chunk from the dataset,
    pred_pan : prediction from the original top image,
    swap_pan : prediction with a batch-neighbour's top image (state unchanged).

Metrics:
- observational regression of pan on cup_x for GT and for the prediction;
- counterfactual regression of dpan = swap_pan - pred_pan on
  dcup_x = cup_x(neighbour) - cup_x(own). A state-only policy gives slope 0;
- direction agreement: share of swap pairs whose pan response points the same
  way the GT regression implies.

Cup centres are read from the CVAT XML annotations (mean of non-outside boxes).
"""

import argparse
import json
import logging
import xml.etree.ElementTree as ET
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


def load_cup_centers(annotation_dir: str, camera_key: str, episodes: list[int]) -> dict[tuple[int, int], float]:
    """Return (episode, frame) -> cup centre x in pixels."""
    camera_dir = Path(annotation_dir) / camera_key.split(".")[-1]
    centers: dict[tuple[int, int], float] = {}
    for episode in episodes:
        path = camera_dir / f"episode_{episode:03d}.xml"
        if not path.is_file():
            raise FileNotFoundError(f"Missing annotation XML: {path}")
        per_frame: dict[int, list[float]] = defaultdict(list)
        for track in ET.parse(path).getroot().findall("track"):
            for box in track.findall("box"):
                if box.get("outside") == "1":
                    continue
                frame = int(box.get("frame"))
                per_frame[frame].append((float(box.get("xtl")) + float(box.get("xbr"))) / 2)
        for frame, xs in per_frame.items():
            centers[(episode, frame)] = sum(xs) / len(xs)
    return centers


def _regression(pairs: list[tuple[float, float]]) -> dict:
    """Least-squares slope and Pearson correlation of y on x."""
    n = len(pairs)
    if n < 2:
        return {"n": n, "slope": None, "pearson": None}
    xs = torch.tensor([p[0] for p in pairs], dtype=torch.float64)
    ys = torch.tensor([p[1] for p in pairs], dtype=torch.float64)
    mx, my = xs.mean(), ys.mean()
    cov = ((xs - mx) * (ys - my)).mean()
    var_x = ((xs - mx) ** 2).mean()
    var_y = ((ys - my) ** 2).mean()
    slope = float(cov / var_x) if var_x > 0 else None
    pearson = float(cov / (var_x * var_y).sqrt()) if var_x > 0 and var_y > 0 else None
    return {"n": n, "slope": slope, "pearson": pearson}


def _spearman(pairs: list[tuple[float, float]]) -> float | None:
    """Rank correlation (no scipy dependency)."""
    n = len(pairs)
    if n < 3:
        return None
    xs = torch.tensor([p[0] for p in pairs], dtype=torch.float64)
    ys = torch.tensor([p[1] for p in pairs], dtype=torch.float64)
    rank_x = xs.argsort().argsort().double()
    rank_y = ys.argsort().argsort().double()
    mx, my = rank_x.mean(), rank_y.mean()
    cov = ((rank_x - mx) * (rank_y - my)).mean()
    var_x = ((rank_x - mx) ** 2).mean()
    var_y = ((rank_y - my) ** 2).mean()
    return float(cov / (var_x * var_y).sqrt()) if var_x > 0 and var_y > 0 else None


def evaluate(args: argparse.Namespace) -> dict:
    checkpoint = Path(args.checkpoint)
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    if cfg.type not in POLICY_CLASSES:
        raise ValueError(f"Unsupported policy type {cfg.type!r} in {checkpoint}")
    if args.camera_key not in cfg.image_features:
        raise ValueError(f"{args.camera_key!r} is not an input camera; available: {list(cfg.image_features)}")

    policy = POLICY_CLASSES[cfg.type].from_pretrained(checkpoint, config=cfg)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(checkpoint))

    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    action_names = metadata.features.get("action", {}).get("names") or []
    pan_index = next((i for i, name in enumerate(action_names) if "shoulder_pan" in name.lower()), 0)
    if not action_names:
        logging.warning("Action names missing from metadata; assuming pan_index=0")
    chunk_size = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))["chunk_size"]

    episodes = parse_episodes(args.episodes)
    cup_centers = load_cup_centers(args.annotation_dir, args.camera_key, episodes)

    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        episodes=episodes,
        delta_timestamps={"action": [i / metadata.fps for i in range(chunk_size)]},
        video_backend=args.dataset_video_backend,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=True,
    )

    records = []
    with torch.no_grad():
        for step, raw_batch in enumerate(dataloader):
            if args.max_batches is not None and step >= args.max_batches:
                break
            batch_episodes = [int(value) for value in raw_batch["episode_index"]]
            batch_frames = [int(value) for value in raw_batch["frame_index"]]
            batch = preprocessor(raw_batch)
            steps = min(args.action_steps or batch["action"].shape[1], batch["action"].shape[1])
            valid = (~batch["action_is_pad"][:, :steps]).float()

            def masked_mean(chunk: torch.Tensor) -> torch.Tensor:
                return (chunk * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)

            gt_pan = masked_mean(batch["action"][:, :steps, pan_index])
            pred_pan = masked_mean(policy.predict_action_chunk(batch)[:, :steps, pan_index])

            swapped_batch = dict(batch)
            swapped_batch[args.camera_key] = torch.roll(batch[args.camera_key], shifts=1, dims=0)
            swap_pan = masked_mean(
                policy.predict_action_chunk(swapped_batch)[:, :steps, pan_index]
            )

            size = len(batch_episodes)
            for i in range(size):
                source = (i - 1) % size  # roll(shifts=1) gives sample i the image of i-1
                records.append(
                    {
                        "episode": batch_episodes[i],
                        "frame": batch_frames[i],
                        "valid_steps": int(valid[i].sum()),
                        "cup_x": cup_centers.get((batch_episodes[i], batch_frames[i])),
                        "cup_x_swapped": cup_centers.get(
                            (batch_episodes[source], batch_frames[source])
                        ),
                        "gt_pan": float(gt_pan[i]),
                        "pred_pan": float(pred_pan[i]),
                        "swap_pan": float(swap_pan[i]),
                    }
                )
            if step % 10 == 0:
                logging.info("step %d (%d samples)", step, len(records))

    if not records:
        raise RuntimeError("No samples were evaluated")

    # --- Observational: pan vs cup_x -------------------------------------
    gt_pairs = [(r["cup_x"], r["gt_pan"]) for r in records if r["cup_x"] is not None and r["valid_steps"] > 0]
    pred_pairs = [(r["cup_x"], r["pred_pan"]) for r in records if r["cup_x"] is not None and r["valid_steps"] > 0]
    observational = {"gt": _regression(gt_pairs), "pred": _regression(pred_pairs)}
    gt_slope = observational["gt"]["slope"]

    # --- Counterfactual: dpan vs dcup_x ----------------------------------
    swap_pairs = [
        (r["cup_x_swapped"] - r["cup_x"], r["swap_pan"] - r["pred_pan"])
        for r in records
        if r["cup_x"] is not None and r["cup_x_swapped"] is not None and r["valid_steps"] > 0
    ]
    counterfactual = _regression(swap_pairs)
    counterfactual["min_delta_cup_x"] = args.min_delta_cup_x

    if gt_slope is not None:
        large = [
            (dcx, dpan) for dcx, dpan in swap_pairs if abs(dcx) >= args.min_delta_cup_x
        ]
        agreement = [
            (dpan > 0) == (gt_slope * dcx > 0) for dcx, dpan in large if dpan != 0
        ]
        counterfactual["direction_agreement"] = (
            sum(agreement) / len(agreement) if agreement else None
        )
        counterfactual["direction_agreement_n"] = len(agreement)
        counterfactual["mean_dpan_when_cup_moved_right"] = (
            sum(dpan for dcx, dpan in large if dcx > 0) / max(sum(1 for dcx, _ in large if dcx > 0), 1)
        )
        counterfactual["mean_dpan_when_cup_moved_left"] = (
            sum(dpan for dcx, dpan in large if dcx < 0) / max(sum(1 for dcx, _ in large if dcx < 0), 1)
        )

    # --- Per-episode table (cup_x is roughly constant within an episode) --
    by_episode: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        by_episode[record["episode"]].append(record)
    per_episode = []
    for episode in sorted(by_episode):
        rows = [r for r in by_episode[episode] if r["cup_x"] is not None and r["valid_steps"] > 0]
        if not rows:
            continue
        per_episode.append(
            {
                "episode": episode,
                "samples": len(rows),
                "cup_x": sum(r["cup_x"] for r in rows) / len(rows),
                "gt_pan": sum(r["gt_pan"] for r in rows) / len(rows),
                "pred_pan": sum(r["pred_pan"] for r in rows) / len(rows),
                "swap_pan": sum(r["swap_pan"] for r in rows) / len(rows),
            }
        )
    cross_episode = {
        "gt": _spearman([(e["cup_x"], e["gt_pan"]) for e in per_episode]),
        "pred": _spearman([(e["cup_x"], e["pred_pan"]) for e in per_episode]),
    }

    return {
        "checkpoint": str(checkpoint),
        "dataset": args.dataset_repo_id,
        "episodes": episodes,
        "camera_key": args.camera_key,
        "protocol": {
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "action_steps": args.action_steps,
            "seed": args.seed,
            "pan_index": pan_index,
            "pan_name": action_names[pan_index] if action_names else None,
        },
        "samples": len(records),
        "samples_with_cup": len(gt_pairs),
        "observational": observational,
        "counterfactual_swap": counterfactual,
        "cross_episode_spearman": cross_episode,
        "per_episode": per_episode,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", default=None)
    parser.add_argument("--dataset.root", dest="dataset_root", default=None)
    parser.add_argument("--episodes", default="7,11,13,14,16,27,35,38,40,43")
    parser.add_argument("--camera-key", default="observation.images.top")
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--min-delta-cup-x", type=float, default=25.0)
    parser.add_argument("--dataset.video_backend", dest="dataset_video_backend", default="pyav")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    checkpoint = Path(args.checkpoint)
    if args.dataset_repo_id is None or args.dataset_root is None:
        inferred = infer_dataset_from_checkpoint(checkpoint)
        if inferred is None:
            raise SystemExit("Cannot infer dataset; pass --dataset.repo_id and --dataset.root")
        args.dataset_repo_id = args.dataset_repo_id or inferred[0]
        args.dataset_root = args.dataset_root or inferred[1]

    result = evaluate(args)

    print("\n===== Pan direction consistency =====")
    print(f"  samples with cup box      {result['samples_with_cup']} / {result['samples']}")
    print(f"  GT  slope (px)            {result['observational']['gt']['slope']}")
    print(f"  pred slope (px)           {result['observational']['pred']['slope']}")
    print(f"  swap slope (px)           {result['counterfactual_swap']['slope']}")
    print(f"  direction agreement       {result['counterfactual_swap'].get('direction_agreement')}")
    print(f"  cross-episode spearman    {result['cross_episode_spearman']}")
    print("  per episode:")
    for row in result["per_episode"]:
        print(
            f"    ep {row['episode']:>3}  cup_x={row['cup_x']:7.1f}  "
            f"gt={row['gt_pan']:+.4f}  pred={row['pred_pan']:+.4f}  swap={row['swap_pan']:+.4f}"
        )

    output = Path(args.output) if args.output else checkpoint / "pan_direction_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
