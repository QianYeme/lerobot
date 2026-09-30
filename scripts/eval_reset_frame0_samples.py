"""Sample full action horizons at reset frame 0 for direction-info diagnosis.

Phase A of the direction-fix plan: for each episode, generate N full action
chunks from the checkpoint at the raw reset frame 0 and store them raw
(postprocessed, degrees). ACT/ACTDet are deterministic so N=1; diffusion draws
a fresh noise prior per sample (global RNG reseeded per sample) so N>1 gives
the model's distribution. Truth/desired actions are NOT stored here; the phase
A analyzer loads them from the original parquet, same convention as
analyze_formal3_h0_bias.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import populate_queues
from lerobot.scripts.offline_eval_act_det import POLICY_CLASSES, collate, parse_episodes
from lerobot.utils.constants import ACTION, OBS_IMAGES


def sample_chunks(policy, cfg, batch, num_samples, seed_base):
    """Return (num_samples, horizon, action_dim) raw chunks for one observation."""
    if cfg.type != "diffusion":
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(batch)
        return chunk.expand(num_samples, -1, -1)
    # The diffusion action queue only keeps n_action_steps entries, so the full
    # sampled horizon cannot be read back from it. Populate the observation
    # queues the same way select_action does, then generate the full chunk
    # directly with a freshly seeded noise prior.
    obs_batch = {k: v for k, v in batch.items() if k != ACTION}
    if policy.config.image_features:
        obs_batch = dict(obs_batch)
        obs_batch[OBS_IMAGES] = torch.stack([obs_batch[key] for key in policy.config.image_features], dim=-4)
    chunks = []
    with torch.inference_mode():
        for i in range(num_samples):
            policy.reset()
            torch.manual_seed(seed_base + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_base + i)
            policy._queues = populate_queues(policy._queues, obs_batch)
            # generate_actions slices [n_obs_steps-1 : n_obs_steps-1+n_action_steps]
            # of the sampled horizon; widen the slice for this diagnostic so the
            # full sampled trajectory is returned, then restore.
            saved = policy.config.n_action_steps
            policy.config.n_action_steps = policy.config.horizon
            try:
                chunks.append(policy.predict_action_chunk(obs_batch))
            finally:
                policy.config.n_action_steps = saved
    return torch.stack(chunks)[:, 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--episodes", default="7,11,13,14,16,27,35,38,40,43")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame-mode", choices=["frame0", "before_S"], default="frame0")
    parser.add_argument("--timetrim-root", type=Path, default=None,
                        help="required when --frame-mode before_S")
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
    state_names = info["features"]["observation.state"]["names"]
    if len(state_names) != cfg.input_features["observation.state"].shape[0]:
        raise ValueError("Dataset / checkpoint state dimensions differ")
    policy = POLICY_CLASSES[cfg.type].from_pretrained(args.checkpoint, config=cfg).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(args.checkpoint),
                                         preprocessor_overrides={"device_processor": {"device": args.device}})
    boundaries = None
    if args.frame_mode == "before_S":
        if args.timetrim_root is None:
            parser.error("--timetrim-root is required with --frame-mode before_S")
        boundaries = {row["episode_index"]: row
                      for row in json.loads((args.timetrim_root / "meta/time_trim_manifest.json").read_text())["boundaries"]}
    report = {"checkpoint": str(args.checkpoint), "dataset": str(args.dataset_root),
              "num_samples": args.num_samples, "frame_mode": args.frame_mode, "episodes": {}}
    for ep in parse_episodes(args.episodes):
        policy.reset()
        dataset = LeRobotDataset(args.repo_id, root=args.dataset_root, episodes=[ep],
                                 delta_timestamps={"action": [i / info["fps"] for i in range(horizon)]},
                                 video_backend="pyav")
        frame_index = 0 if args.frame_mode == "frame0" else max(0, boundaries[ep]["start"] - 1)
        batch = next(iter(DataLoader(Subset(dataset, range(frame_index, frame_index + 1)), batch_size=1,
                                     collate_fn=collate, num_workers=0)))
        raw_state = batch["observation.state"][0].numpy().copy()
        raw_truth = batch["action"][0].numpy().copy()
        batch = pre(batch)
        chunks = sample_chunks(policy, cfg, batch, args.num_samples, seed_base=1000 + ep * 100)
        raw = post(chunks.reshape(-1, chunks.shape[-1])).reshape(chunks.shape).cpu().numpy()
        if not np.isfinite(raw).all():
            raise ValueError("Nonfinite action predictions")
        np.savez_compressed(args.output / f"episode_{ep:03d}_samples.npz",
                            samples_raw=raw, truth_raw=raw_truth, state=raw_state)
        report["episodes"][str(ep)] = {"samples": raw.shape[0], "horizon": raw.shape[1],
                                       "action_dim": raw.shape[2]}
        print(f"episode={ep} samples={raw.shape[0]} horizon={raw.shape[1]}", flush=True)
    (args.output / "sampling.done").write_text("exit=0\n", encoding="utf-8")
    (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
