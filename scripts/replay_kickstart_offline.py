"""B2 diagnostic: kick-start offline replay with the existing DET 100k model.

Zero training cost. Answers whether executing the first-motion row r* of the
predicted chunk (a "kick-start") lets the 100k model produce a usable trajectory
from the raw reset frame 0, and whether the model takes over at row 0 once the
state leaves the stationary manifold.

Two modes, both teacher-free (the model sees only recorded observations; its
executed actions are never fed back):

  A) openloop-rstar: at raw frame 0, execute the predicted chunk from its first
     motion row r* onward (open loop, no replanning). Truth = original actions
     S..S+L-1 (S = time-trim start, L = min(horizon - r*, max_steps)). Shows
     what trajectory the model's own plan produces from r*.

  B) kickstart-hybrid: closed loop over recorded observations with a hybrid
     clock. At each observation, predict a fresh chunk; if row 0 deviates less
     than MOTION_DEG from the baseline (stationary plan), fire the kick: execute
     chunk row r* (its first motion row). Clock rule: the first fire advances
     the clock to S+1 (the kick replicates the recorded first motion, so the
     post-kick world state is proxied by the recorded frame right after it);
     later fires and normal steps advance the clock by 1. Truth alignment: the
     fired action is compared to action[S], subsequent executed actions to
     actions[S+1..]. Shows how many kicks are needed and whether the model
     takes over at row 0 on moving observations.

Conventions match analyze_phaseA_direction_info.py: baseline = median of the
episode's first 15 original actions; desired first motion = action[S] - baseline;
arm = first 5 action dims; MOTION_DEG = 5.0. All values natively degrees.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Subset

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.offline_eval_act_det import POLICY_CLASSES, collate, parse_episodes

ARM = 5
MOTION_DEG = 5.0


def load_original_actions(original_root: Path, episode: int) -> np.ndarray:
    table = pq.read_table(original_root / "data/chunk-000/file-000.parquet",
                          columns=["episode_index", "frame_index", "action"])
    rows = table.filter(pc.equal(table["episode_index"], episode)).sort_by("frame_index")
    return np.stack(rows["action"].to_numpy())


def first_motion_row(chunk: np.ndarray, baseline: np.ndarray):
    """First chunk row with arm L-inf deviation >= MOTION_DEG; None if absent."""
    for row in range(chunk.shape[0]):
        if np.max(np.abs(chunk[row, :ARM] - baseline[:ARM])) >= MOTION_DEG:
            return row
    return None


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else np.nan


def obs_batch(dataset: LeRobotDataset, frame: int, pre) -> dict:
    batch = next(iter(DataLoader(Subset(dataset, range(frame, frame + 1)), batch_size=1,
                                 collate_fn=collate, num_workers=0)))
    return pre(batch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="original fit48 dataset root (kind_merged_nomaster_fit48)")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--timetrim-root", type=Path, required=True,
                        help="time-trim dataset root; provides meta/time_trim_manifest.json")
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--mode", choices=["a", "b", "both"], default="both")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    info = json.loads((args.dataset_root / "meta/info.json").read_text())
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.device = args.device
    horizon = getattr(cfg, "chunk_size", None) or getattr(cfg, "horizon", None)
    if not horizon:
        raise ValueError(f"Unsupported action horizon for policy type {cfg.type!r}")
    policy = POLICY_CLASSES[cfg.type].from_pretrained(args.checkpoint, config=cfg).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(args.checkpoint),
                                         preprocessor_overrides={"device_processor": {"device": args.device}})
    boundaries = {row["episode_index"]: row
                  for row in json.loads((args.timetrim_root / "meta/time_trim_manifest.json").read_text())["boundaries"]}

    episodes = parse_episodes(args.episodes)
    per_episode = {}
    for ep in episodes:
        policy.reset()
        dataset = LeRobotDataset(args.repo_id, root=args.dataset_root, episodes=[ep],
                                 delta_timestamps={"action": [i / info["fps"] for i in range(horizon)]},
                                 video_backend="pyav")
        actions = load_original_actions(args.dataset_root, ep)
        num_frames = actions.shape[0]
        start = boundaries[ep]["start"]
        baseline = np.median(actions[0:15], axis=0)
        desired = actions[start][:ARM] - baseline[:ARM]
        entry: dict = {"S": start, "frames": num_frames}

        batch0 = obs_batch(dataset, 0, pre)
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(batch0)
        chunk0 = post(chunk.reshape(-1, chunk.shape[-1])).reshape(chunk.shape).cpu().numpy()[0]
        r_star = first_motion_row(chunk0, baseline)
        entry["frame0_r_star"] = r_star

        if args.mode in ("a", "both") and r_star is not None:
            length = min(horizon - r_star, args.max_steps)
            executed = chunk0[r_star:r_star + length]
            truth = actions[start:start + length]
            entry["mode_a"] = {
                "r_star": r_star, "length": length,
                "first_cos": cosine(executed[0, :ARM] - baseline[:ARM], desired),
                "arm_mae": float(np.mean(np.abs(executed[:, :ARM] - truth[:, :ARM]))),
                "gripper_mae": float(np.mean(np.abs(executed[:, ARM:] - truth[:, ARM:]))),
                "norm_ratio": float(np.linalg.norm(executed[0, :ARM] - baseline[:ARM]) /
                                    np.linalg.norm(desired)) if np.linalg.norm(desired) else np.nan,
            }

        if args.mode in ("b", "both"):
            executed, truth_ref = [], []
            n_fires, fired_first, t = 0, False, 0
            takeover = None
            for _ in range(args.max_steps):
                if t >= num_frames - horizon:
                    break
                batch = obs_batch(dataset, t, pre)
                with torch.inference_mode():
                    c = policy.predict_action_chunk(batch)
                raw = post(c.reshape(-1, c.shape[-1])).reshape(c.shape).cpu().numpy()[0]
                row0_delta = np.max(np.abs(raw[0, :ARM] - baseline[:ARM]))
                if row0_delta < MOTION_DEG:
                    r = first_motion_row(raw, baseline)
                    if r is None:
                        executed.append(raw[0])
                        truth_ref.append(actions[t])
                        t += 1
                        continue
                    n_fires += 1
                    executed.append(raw[r])
                    if not fired_first:
                        truth_ref.append(actions[start])
                        fired_first = True
                        t = start + 1
                    else:
                        truth_ref.append(actions[t])
                        t += 1
                else:
                    if takeover is None:
                        takeover = float(row0_delta)
                    executed.append(raw[0])
                    truth_ref.append(actions[t])
                    t += 1
            executed = np.stack(executed)
            truth = np.stack(truth_ref)
            entry["mode_b"] = {
                "steps": executed.shape[0], "n_fires": n_fires,
                "first_fire_cos": cosine(executed[0, :ARM] - baseline[:ARM], desired)
                if executed.shape[0] else np.nan,
                "arm_mae": float(np.mean(np.abs(executed[:, :ARM] - truth[:, :ARM]))),
                "gripper_mae": float(np.mean(np.abs(executed[:, ARM:] - truth[:, ARM:]))),
                "takeover_row0_delta": takeover,
            }

        per_episode[str(ep)] = entry
        print(f"episode={ep} done: {json.dumps(entry, ensure_ascii=False)}", flush=True)

    report: dict = {"checkpoint": str(args.checkpoint), "dataset": str(args.dataset_root),
                    "mode": args.mode, "max_steps": args.max_steps,
                    "episodes": episodes, "per_episode": per_episode}
    if args.mode in ("a", "both"):
        a = [v["mode_a"] for v in per_episode.values() if "mode_a" in v]
        if a:
            report["mode_a_summary"] = {
                "n": len(a),
                "first_cos_positive": int(np.sum([e["first_cos"] > 0 for e in a])),
                "first_cos_mean": float(np.nanmean([e["first_cos"] for e in a])),
                "arm_mae_mean": float(np.nanmean([e["arm_mae"] for e in a])),
                "gripper_mae_mean": float(np.nanmean([e["gripper_mae"] for e in a])),
                "norm_ratio_median": float(np.nanmedian([e["norm_ratio"] for e in a])),
            }
    if args.mode in ("b", "both"):
        b = [v["mode_b"] for v in per_episode.values() if "mode_b" in v]
        if b:
            report["mode_b_summary"] = {
                "n": len(b),
                "n_fires_mean": float(np.nanmean([e["n_fires"] for e in b])),
                "first_fire_cos_positive": int(np.sum([e["first_fire_cos"] > 0 for e in b])),
                "first_fire_cos_mean": float(np.nanmean([e["first_fire_cos"] for e in b])),
                "arm_mae_mean": float(np.nanmean([e["arm_mae"] for e in b])),
                "gripper_mae_mean": float(np.nanmean([e["gripper_mae"] for e in b])),
                "takeover_mean_delta": float(np.nanmean([e["takeover_row0_delta"] or np.nan for e in b])),
                "takeover_ok": int(np.sum([(e["takeover_row0_delta"] or 0) >= MOTION_DEG for e in b])),
            }
    (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    lines = [f"# B2 kick-start replay: {args.output.name}", "",
             f"- checkpoint: {report['checkpoint']} | mode: {args.mode} | max_steps: {args.max_steps}",
             f"- episodes: {len(episodes)}"]
    if "mode_a_summary" in report:
        m = report["mode_a_summary"]
        lines += ["", "## Mode A: open-loop from r* (frame 0)",
                  f"- first-action cosine+ {m['first_cos_positive']}/{m['n']} | mean {m['first_cos_mean']:.3f}",
                  f"- arm MAE {m['arm_mae_mean']:.2f}° | gripper MAE {m['gripper_mae_mean']:.2f}° | norm ratio median {m['norm_ratio_median']:.3f}"]
    if "mode_b_summary" in report:
        m = report["mode_b_summary"]
        lines += ["", "## Mode B: kick-start hybrid clock",
                  f"- fires per episode mean {m['n_fires_mean']:.2f} | first-fire cosine+ {m['first_fire_cos_positive']}/{m['n']} | mean {m['first_fire_cos_mean']:.3f}",
                  f"- arm MAE {m['arm_mae_mean']:.2f}° | gripper MAE {m['gripper_mae_mean']:.2f}°",
                  f"- takeover at first moving obs: {m['takeover_ok']}/{m['n']} (row0 delta mean {m['takeover_mean_delta']:.1f}°)"]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
