"""Paired h0 action evaluation with original versus raw-zero master gripper state.

This is teacher-forced offline evaluation, NOT a closed-loop robot rollout.
Only master_gripper.pos is changed, before the saved checkpoint preprocessor.
Phase intervals are manually reviewed, half-open episode-local frame ranges.
"""

import argparse
import csv
import hashlib
import json
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.offline_eval_act_det import POLICY_CLASSES, collate, parse_episodes

PHASES = {"approach", "grasp_transport", "place", "other", "unknown"}


def zero_master_state(raw_batch: dict, master_index: int) -> dict:
    result = dict(raw_batch)
    result["observation.state"] = raw_batch["observation.state"].clone()
    result["observation.state"][..., master_index] = 0.0
    return result


def load_phases(path: Path, lengths: dict[int, int]) -> dict[tuple[int, int], str]:
    annotations = json.loads(path.read_text(encoding="utf-8"))
    lookup = {}
    for episode, length in lengths.items():
        if str(episode) not in annotations["episodes"]:
            raise ValueError(f"Phase file is missing episode {episode}")
        entry = annotations["episodes"][str(episode)]
        for segment in entry["segments"]:
            start, stop, phase = segment["start"], segment["stop"], segment["phase"]
            if phase not in PHASES or not (0 <= start < stop <= length):
                raise ValueError(f"Invalid segment in episode {episode}: {segment}")
            if phase != "unknown" and not entry.get("reviewed", False):
                raise ValueError(f"Episode {episode} has unreviewed non-unknown phase labels")
            for frame in range(start, stop):
                key = (episode, frame)
                if key in lookup:
                    raise ValueError(f"Overlapping phase intervals at {key}")
                lookup[key] = phase
    return lookup


def summarize_h0(original, zero, target, action_names) -> dict:
    if len(target) == 0:
        return {"frames": 0}
    errors_original = (original - target).abs().double().mean(dim=0)
    errors_zero = (zero - target).abs().double().mean(dim=0)
    action_delta = (zero - original).abs().double().mean(dim=0)
    groups = {
        "arm": [i for i, name in enumerate(action_names) if "gripper" not in name],
        "gripper": [i for i, name in enumerate(action_names) if "gripper" in name],
        "all_unweighted": list(range(len(action_names))),
    }

    def metrics(indices):
        baseline = float(errors_original[indices].mean())
        ablated = float(errors_zero[indices].mean())
        return {
            "original_l1": baseline,
            "zero_master_l1": ablated,
            "error_increase": ablated - baseline,
            "zero_vs_original_delta_l1": float(action_delta[indices].mean()),
        }

    return {
        "frames": len(target),
        "by_group": {name: metrics(indices) for name, indices in groups.items() if indices},
        "by_action_dim": {name: metrics([i]) for i, name in enumerate(action_names)},
    }


def evaluate(args) -> dict:
    started = time.perf_counter()
    checkpoint, output = Path(args.checkpoint), Path(args.output)
    csv_path = output.with_suffix(".csv")
    if output.exists() or csv_path.exists():
        raise FileExistsError(f"Output already exists: {output} or {csv_path}; use a new output path")
    if args.batch_size < 1 or (args.max_batches is not None and args.max_batches < 1):
        raise ValueError("batch-size and max-batches must be positive")
    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    state_names = metadata.features["observation.state"]["names"]
    action_names = metadata.features["action"]["names"]
    master_index = state_names.index("master_gripper.pos")
    episodes = parse_episodes(args.episodes)
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    if cfg.type not in POLICY_CLASSES:
        raise ValueError(f"Unsupported policy {cfg.type}")
    # Labels are not consumed by action-only inference; do not change feature injection.
    if cfg.type == "act_det":
        cfg.annotation_dir = None
        cfg.mask_dir = None
    cfg.pretrained_path = checkpoint
    policy = POLICY_CLASSES[cfg.type].from_pretrained(checkpoint, config=cfg)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=str(checkpoint)
    )
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        episodes=episodes,
        delta_timestamps={"action": [i / metadata.fps for i in range(cfg.chunk_size)]},
        video_backend="pyav",
    )
    lengths = {ep: int(metadata.episodes[ep]["length"]) for ep in episodes}
    phases = load_phases(Path(args.phase_file), lengths)
    expected_frames = sum(lengths.values())
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=collate, drop_last=False,
    )
    predictions = {key: [] for key in ("original", "zero", "target", "original_raw", "zero_raw", "target_raw")}
    frame_episodes, frame_phases, keys, master_values, normalized_master_values = [], [], [], [], []
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["episode", "frame", "phase", "master_original_raw", "master_zero_raw"]
    fields += [f"{kind}.{name}" for kind in ("target", "original", "zero_master") for name in action_names]
    batches = 0
    with csv_path.open("x", newline="", encoding="utf-8") as csv_file, torch.inference_mode():
        writer = csv.writer(csv_file)
        writer.writerow(fields)
        for batch_index, raw in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch_episodes = raw["episode_index"].tolist()
            batch_frames = raw["frame_index"].tolist()
            original_master = raw["observation.state"][..., master_index].clone()
            raw_zero = zero_master_state(raw, master_index)
            # Saved processors are reused and reset between independent conditions.
            preprocessor.reset()
            original_batch = preprocessor(dict(raw))
            preprocessor.reset()
            zero_batch = preprocessor(raw_zero)
            torch.testing.assert_close(original_batch["action"], zero_batch["action"], rtol=0, atol=0)
            other_indices = [i for i in range(len(state_names)) if i != master_index]
            torch.testing.assert_close(
                original_batch["observation.state"][..., other_indices],
                zero_batch["observation.state"][..., other_indices], rtol=0, atol=0,
            )
            for camera in cfg.image_features:
                torch.testing.assert_close(original_batch[camera], zero_batch[camera], rtol=0, atol=0)
            original = policy.predict_action_chunk(original_batch)[:, 0]
            zero = policy.predict_action_chunk(zero_batch)[:, 0]
            target = original_batch["action"][:, 0]
            postprocessor.reset()
            original_raw = postprocessor(original).cpu()
            postprocessor.reset()
            zero_raw = postprocessor(zero).cpu()
            valid = ~original_batch["action_is_pad"][:, 0].cpu()
            values = {
                "original": original.cpu(), "zero": zero.cpu(), "target": target.cpu(),
                "original_raw": original_raw, "zero_raw": zero_raw, "target_raw": raw["action"][:, 0].cpu(),
            }
            for key, value in values.items():
                if not torch.isfinite(value).all():
                    raise RuntimeError(f"Non-finite {key} at batch {batch_index}")
                predictions[key].append(value[valid])
            for i, (ep, frame) in enumerate(zip(batch_episodes, batch_frames, strict=True)):
                if not valid[i]:
                    continue
                phase = phases.get((ep, frame), "unknown")
                keys.append((ep, frame))
                frame_episodes.append(ep)
                frame_phases.append(phase)
                master_values.append(float(original_master[i].reshape(-1)[-1]))
                normalized_master_values.append(float(zero_batch["observation.state"][i, ..., master_index].reshape(-1)[-1]))
                writer.writerow(
                    [ep, frame, phase, master_values[-1], 0.0]
                    + values["target_raw"][i].tolist()
                    + original_raw[i].tolist() + zero_raw[i].tolist()
                )
            batches += 1
            if batch_index % 25 == 0:
                csv_file.flush()
                logging.info("batch %d, frames %d/%d, elapsed %.1fs", batch_index, len(keys), expected_frames, time.perf_counter() - started)
    if not keys or len(set(keys)) != len(keys):
        raise RuntimeError("No valid frames or duplicate frame identities")
    complete = len(keys) == expected_frames
    if args.max_batches is None and not complete:
        raise RuntimeError(f"Incomplete full evaluation: {len(keys)}/{expected_frames}")
    predictions = {key: torch.cat(values) for key, values in predictions.items()}

    def summary(mask):
        normalized = summarize_h0(
            predictions["original"][mask], predictions["zero"][mask], predictions["target"][mask], action_names,
        )
        raw = summarize_h0(
            predictions["original_raw"][mask], predictions["zero_raw"][mask], predictions["target_raw"][mask], action_names,
        )
        return {"normalized": normalized, "raw_units_by_action_dim": raw.get("by_action_dim", {})}

    result = {
        "checkpoint": str(checkpoint), "dataset": args.dataset_repo_id, "episodes": episodes,
        "protocol": {"horizon": 0, "batch_size": args.batch_size, "max_batches": args.max_batches,
                     "n_action_steps": cfg.n_action_steps, "temporal_ensemble_coeff": cfg.temporal_ensemble_coeff,
                     "master_index": master_index, "raw_zero_before_preprocessor": True,
                     "metric_weighting": "unweighted; arm=5 joints, gripper=1; no gripper x3 weighting",
                     "teacher_forced_state": True, "video_backend": "pyav", "num_workers": 0},
        "state_names": state_names, "action_names": action_names,
        "frames": len(keys), "expected_frames": expected_frames, "complete": complete, "batches": batches,
        "master_original_raw_mean": sum(master_values) / len(master_values),
        "master_zero_normalized_min_max": [min(normalized_master_values), max(normalized_master_values)],
        "phase_file_sha256": hashlib.sha256(Path(args.phase_file).read_bytes()).hexdigest(),
        "phase_annotations": json.loads(Path(args.phase_file).read_text(encoding="utf-8")),
        "overall": summary(torch.ones(len(keys), dtype=torch.bool)),
        "by_phase": {phase: summary(torch.tensor([p == phase for p in frame_phases])) for phase in sorted(PHASES)},
        "by_episode": {str(ep): summary(torch.tensor([e == ep for e in frame_episodes])) for ep in episodes},
        "elapsed_seconds": time.perf_counter() - started, "peak_cuda_memory_allocated": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "per_frame_csv": str(csv_path),
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", default="QYyyyyyyy/formal1_C")
    parser.add_argument("--dataset.root", dest="dataset_root", default="数据集/formal1_C")
    parser.add_argument("--episodes", default="7,11,13,14,16,27,35,38,40,43")
    parser.add_argument("--phase-file", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = evaluate(args)
    print(json.dumps({"frames": result["frames"], "complete": result["complete"], "overall": result["overall"]}, indent=2))


if __name__ == "__main__":
    main()
