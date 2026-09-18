"""Capture chronological ACT chunks and evaluate target continuity without moving a robot."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.offline_eval_act_det import POLICY_CLASSES, collate, parse_episodes


def group_mean(values):
    values = np.asarray(values)
    if not values.size:
        return None
    mean = values.reshape(-1, values.shape[-1]).mean(axis=0)
    return {"per_joint": mean.tolist(), "arm": float(mean[:-1].mean()), "gripper": float(mean[-1])}


def queue_replay(chunks, steps):
    if steps < 1 or steps > chunks.shape[1]:
        raise ValueError("Execution length must be within chunk size")
    return np.stack([chunks[(t // steps) * steps, t % steps] for t in range(len(chunks))])


def ensemble_replay(chunks, coefficient):
    """Direct diagonal calculation; positive coefficient favors earliest prediction origin."""
    output = []
    horizon = chunks.shape[1]
    for t in range(len(chunks)):
        origins = np.arange(max(0, t - horizon + 1), t + 1)
        candidates = chunks[origins, t - origins]
        weights = np.exp(-coefficient * np.arange(len(origins)))
        output.append((candidates * weights[:, None]).sum(axis=0) / weights.sum())
    return np.asarray(output)


def trajectory_metrics(target, truth):
    return {"mae": group_mean(np.abs(target - truth)),
            "step_delta": group_mean(np.abs(np.diff(target, axis=0))),
            "second_delta": group_mean(np.abs(np.diff(target, n=2, axis=0))),
            "demonstration_step_delta": group_mean(np.abs(np.diff(truth, axis=0))),
            "motion_std_per_joint": target.std(axis=0).tolist()}


def sequence_metrics(chunks, truth, valid):
    n, horizon, _ = chunks.shape
    errors = np.abs(chunks - truth)
    metrics = {"frames": n, "h0_mae": group_mean(errors[:, 0]), "horizon_mae": {}, "chunk_conflict": {}}
    for end in (5, 10, 30, 100):
        stop = min(end, horizon)
        metrics["horizon_mae"][f"first_{stop}"] = group_mean(errors[:, :stop][valid[:, :stop]])
    conflicts = np.abs(chunks[:-1, 1:] - chunks[1:, :-1])
    usable = valid[:-1, 1:] & valid[1:, :-1]
    for lo, hi in ((1, 5), (6, 10), (11, 30), (31, 99)):
        stop = min(hi, horizon - 1)
        if stop < lo:
            continue
        mask = usable[:, lo - 1:stop]
        metrics["chunk_conflict"][f"h{lo}_{stop}"] = {
            "pairs": int(mask.sum()), "mean": group_mean(conflicts[:, lo - 1:stop][mask])}
    gt = truth[:, 0]
    metrics["execution_replay"] = {}
    for steps in (1, 5, 10):
        if steps <= horizon:
            metrics["execution_replay"][f"queue_{steps}"] = trajectory_metrics(queue_replay(chunks, steps), gt)
    for coefficient in (0.01, 0.03, 0.1):
        metrics["execution_replay"][f"ensemble_{coefficient}"] = trajectory_metrics(ensemble_replay(chunks, coefficient), gt)
    return metrics


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--episodes", default="7,11,13,14,16,27,35,38,40,43")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=None, help="Per-episode prefix for smoke tests only")
    parser.add_argument("--raw-master-zero", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    info = json.loads((args.dataset_root / "meta/info.json").read_text())
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.device = args.device
    state_names = info["features"]["observation.state"]["names"]
    if len(state_names) != cfg.input_features["observation.state"].shape[0]:
        raise ValueError("Dataset / checkpoint state dimensions differ")
    if args.raw_master_zero and "master_gripper.pos" not in state_names:
        raise ValueError("Raw-master-zero requested for dataset without this field")
    policy = POLICY_CLASSES[cfg.type].from_pretrained(args.checkpoint, config=cfg).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(args.checkpoint),
                                       preprocessor_overrides={"device_processor": {"device": args.device}})
    args.output.mkdir(parents=True)
    report = {"checkpoint": str(args.checkpoint), "dataset": str(args.dataset_root),
              "state_names": state_names, "raw_master_zero": args.raw_master_zero,
              "interpretation": "teacher-forced chronological replay; not closed-loop success or physical jerk",
              "max_frames_per_episode": args.max_frames, "device": args.device,
              "timing": "batch=1; data wait, preprocessor, synchronized network, postprocessor; warmup excluded",
              "episodes": {}}
    for ep in parse_episodes(args.episodes):
        policy.reset()
        dataset = LeRobotDataset(args.repo_id, root=args.dataset_root, episodes=[ep],
                                 delta_timestamps={"action": [i / info["fps"] for i in range(cfg.chunk_size)]},
                                 video_backend="pyav")
        length = min(len(dataset), args.max_frames) if args.max_frames is not None else len(dataset)
        if length < 2:
            raise ValueError("At least two consecutive frames are required")
        loader = iter(DataLoader(Subset(dataset, range(length)), batch_size=1, collate_fn=collate, num_workers=0))
        predicted, normalized_truth, raw_predicted, raw_truth, validity, states, timings = [], [], [], [], [], [], []
        with torch.inference_mode():
            for frame in range(length):
                started = time.perf_counter()
                batch = next(loader)
                fetched = time.perf_counter()
                if int(batch["episode_index"][0]) != ep or int(batch["frame_index"][0]) != frame:
                    raise ValueError("Dataset samples are not contiguous within episode")
                states.append(batch["observation.state"][0].numpy().copy())
                raw_truth.append(batch["action"][0].numpy().copy())
                validity.append((~batch["action_is_pad"][0]).numpy())
                if args.raw_master_zero:
                    batch["observation.state"][:, state_names.index("master_gripper.pos")] = 0
                batch = pre(batch)
                sync(args.device)
                processed = time.perf_counter()
                if frame == 0:
                    first = policy.predict_action_chunk(batch)
                    repeated = policy.predict_action_chunk(batch)
                    torch.testing.assert_close(first, repeated, rtol=1e-5, atol=1e-6)
                    for _ in range(3):
                        policy.predict_action_chunk(batch)
                    sync(args.device)
                    processed = time.perf_counter()
                chunk = policy.predict_action_chunk(batch)
                sync(args.device)
                inferred = time.perf_counter()
                predicted.append(chunk[0].cpu().numpy())
                normalized_truth.append(batch["action"][0].cpu().numpy())
                raw_predicted.append(post(chunk.reshape(-1, chunk.shape[-1])).reshape(chunk.shape)[0].cpu().numpy())
                finished = time.perf_counter()
                if frame > 0:
                    timings.append([fetched - started, processed - fetched, inferred - processed, finished - inferred])
                if frame % 100 == 0:
                    print(f"episode={ep} frame={frame}/{length}", flush=True)
        pred, gt, raw_pred, raw_gt = map(np.asarray, (predicted, normalized_truth, raw_predicted, raw_truth))
        valid = np.asarray(validity)
        if not np.isfinite(pred).all() or not np.isfinite(raw_pred).all():
            raise ValueError("Nonfinite action predictions")
        np.savez_compressed(args.output / f"episode_{ep:03d}_chunks.npz", prediction=pred, truth=gt,
                            prediction_raw=raw_pred, truth_raw=raw_gt, valid=valid, state=np.asarray(states))
        episode_report = {"normalized": sequence_metrics(pred, gt, valid), "raw_units": sequence_metrics(raw_pred, raw_gt, valid)}
        episode_report["timing_ms"] = {name: dict(zip(("p50", "p95", "p99"),
                                                     np.percentile(np.asarray(timings)[:, index] * 1000, [50, 95, 99]).tolist()))
                                       for index, name in enumerate(("data_wait", "preprocessor", "network", "postprocessor"))}
        report["episodes"][str(ep)] = episode_report
        (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"episode": ep, "h0": episode_report["normalized"]["h0_mae"],
                          "network_ms": episode_report["timing_ms"]["network"]}), flush=True)
    (args.output / "evaluation.done").write_text("exit=0\n", encoding="utf-8")


if __name__ == "__main__":
    main()
