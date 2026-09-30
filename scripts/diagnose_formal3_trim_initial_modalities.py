"""Localize the original-frame-0 vs time-trim-start deployment shift by modality swaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.offline_eval_act_det import POLICY_CLASSES, collate, parse_episodes
from lerobot.scripts.offline_eval_act_sequence import predict_chunk


STATE = "observation.state"
TOP = "observation.images.top"
GRIPPER = "observation.images.gripper"


def clone_batch(batch: dict) -> dict:
    return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def metrics(prediction: np.ndarray, truth: np.ndarray, threshold: float) -> dict:
    baseline = np.median(truth[:15], axis=0)
    moved = np.max(np.abs(truth[:, :5] - baseline[None, :5]), axis=1) >= threshold
    indices = np.flatnonzero(moved)
    if not len(indices):
        raise ValueError("Ground-truth chunk has no departure")
    departure = int(indices[0])
    desired = truth[departure, :5]
    predicted = prediction[0, :5]
    desired_delta = desired - baseline[:5]
    predicted_delta = predicted - baseline[:5]
    denominator = float(np.linalg.norm(desired_delta) * np.linalg.norm(predicted_delta))
    return {
        "gt_departure_h": departure,
        "arm_mae_to_first_gt_motion": float(np.abs(predicted - desired).mean()),
        "motion_cosine": float(np.dot(predicted_delta, desired_delta) / denominator) if denominator else None,
        "motion_norm_ratio": float(np.linalg.norm(predicted_delta) / np.linalg.norm(desired_delta)),
        "pan_direction_ok": bool(np.sign(predicted_delta[0]) == np.sign(desired_delta[0])),
        "prediction_h0_arm": predicted.tolist(),
        "desired_arm": desired.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--trimmed-root", type=Path, required=True)
    parser.add_argument("--original-repo-id", required=True)
    parser.add_argument("--trimmed-repo-id", required=True)
    parser.add_argument("--episodes", default="9,11,17,25,26,28,30,31,43,52,55,59")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=5.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.device = args.device
    horizon = cfg.chunk_size
    fps = json.loads((args.original_root / "meta/info.json").read_text())["fps"]
    delta = {"action": [index / fps for index in range(horizon)]}
    policy = POLICY_CLASSES[cfg.type].from_pretrained(args.checkpoint, config=cfg).eval()
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    variants = {
        "original_all": (),
        "trimmed_all": (STATE, TOP, GRIPPER),
        "original_plus_trimmed_state": (STATE,),
        "original_plus_trimmed_top": (TOP,),
        "original_plus_trimmed_gripper": (GRIPPER,),
        "original_plus_trimmed_images": (TOP, GRIPPER),
        "trimmed_plus_original_state": (TOP, GRIPPER),
        "trimmed_plus_original_top": (STATE, GRIPPER),
        "trimmed_plus_original_gripper": (STATE, TOP),
    }
    results = {name: [] for name in variants}
    per_episode = {}
    with torch.inference_mode():
        for episode in parse_episodes(args.episodes):
            original = LeRobotDataset(
                args.original_repo_id, root=args.original_root, episodes=[episode],
                delta_timestamps=delta, video_backend="pyav",
            )
            trimmed = LeRobotDataset(
                args.trimmed_repo_id, root=args.trimmed_root, episodes=[episode],
                delta_timestamps=delta, video_backend="pyav",
            )
            original_batch = collate([original[0]])
            trimmed_batch = collate([trimmed[0]])
            truth = original_batch["action"][0].numpy()
            episode_result = {}
            for name, trimmed_keys in variants.items():
                batch = clone_batch(original_batch)
                for key in trimmed_keys:
                    batch[key] = trimmed_batch[key].clone()
                policy.reset()
                processed = pre(batch)
                chunk = predict_chunk(policy, cfg, processed)
                raw = post(chunk.reshape(-1, chunk.shape[-1])).reshape(chunk.shape)[0].cpu().numpy()
                item = metrics(raw, truth, args.threshold)
                episode_result[name] = item
                results[name].append(item)
            per_episode[str(episode)] = episode_result

    summary = {}
    for name, items in results.items():
        summary[name] = {
            "arm_mae_mean": float(np.mean([item["arm_mae_to_first_gt_motion"] for item in items])),
            "cosine_mean": float(np.mean([item["motion_cosine"] for item in items])),
            "cosine_positive_rate": float(np.mean([item["motion_cosine"] > 0 for item in items])),
            "norm_ratio_median": float(np.median([item["motion_norm_ratio"] for item in items])),
            "pan_direction_rate": float(np.mean([item["pan_direction_ok"] for item in items])),
        }
    report = {
        "checkpoint": str(args.checkpoint),
        "episodes": parse_episodes(args.episodes),
        "threshold_degrees": args.threshold,
        "interpretation": "teacher-forced modality-swap diagnostic; not closed-loop or real-robot evidence",
        "summary": summary,
        "per_episode": per_episode,
    }
    args.output.mkdir(parents=True)
    (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Formal3 initial-observation modality swap",
        "",
        "| variant | first-motion arm MAE | cosine | cosine > 0 | norm ratio | pan direction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in summary.items():
        lines.append(
            f"| {name} | {item['arm_mae_mean']:.4f} | {item['cosine_mean']:.4f} | "
            f"{item['cosine_positive_rate']:.2f} | {item['norm_ratio_median']:.3f} | "
            f"{item['pan_direction_rate']:.2f} |"
        )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
