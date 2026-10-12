#!/usr/bin/env python
"""T0.2 frozen OC-ACT token-transplant directionality diagnostic.

This is a Dev-only, no-training intervention.  Pair selection is completed
before any transplant action deltas are computed and is recorded verbatim.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.offline_eval_act_det import collate, parse_episodes
from lerobot.utils.constants import ACTION, OBS_IMAGES


DEV_EPISODES = [3, 5, 12, 15, 21, 28, 33, 39, 41, 47, 53, 58]
X_DIM = 0
Y_DIM = 2
EXPERT_X_CORR = -0.26654395
EXPERT_Y_CORR = 0.22405419
CONFIDENCE_MIN = 0.20
PHASE_MAX_FRACTION = 0.75
SIGN_EPS = 1e-6


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return pearson(rx, ry)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def action_h0(pred: torch.Tensor) -> torch.Tensor:
    return pred[:, 0] if pred.ndim == 3 else pred


def build_pair_candidates(records: list[dict], seed: int) -> tuple[dict, dict]:
    """Freeze pair selection using only metadata, before action deltas."""
    rng = np.random.default_rng(seed)
    valid = [r for r in records if r["eligible"]]
    if len(valid) < 2:
        raise RuntimeError(f"Only {len(valid)} eligible frames; cannot construct pairs")
    xs = np.asarray([r["x"] for r in valid], dtype=float)
    ys = np.asarray([r["y"] for r in valid], dtype=float)
    x_span = float(np.max(xs) - np.min(xs))
    y_span = float(np.max(ys) - np.min(ys))
    # State standardisation is frozen before any counterfactual forward pass.
    states = np.asarray([r["state"] for r in valid], dtype=float)
    scale = np.std(states, axis=0)
    scale[scale < 1e-6] = 1.0
    z = states / scale
    times = np.asarray([r["time_fraction"] for r in valid], dtype=float)

    def matched_pairs(axis: int, minimum_delta: float, maximum: int | None = None) -> list[dict]:
        values = xs if axis == 0 else ys
        order = np.argsort(values)
        out: list[dict] = []
        used: set[tuple[int, int]] = set()
        for pos in order:
            remaining = np.arange(len(valid))
            deltas = np.abs(values[remaining] - values[pos])
            candidates = remaining[(deltas >= minimum_delta) & (remaining != pos)]
            if len(candidates) == 0:
                continue
            phase_ok = np.abs(times[candidates] - times[pos]) <= 0.20
            candidates = candidates[phase_ok]
            if len(candidates) == 0:
                continue
            dist = np.linalg.norm(z[candidates] - z[pos], axis=1)
            donor = int(candidates[np.argmin(dist)])
            key = (int(pos), donor)
            if key in used:
                continue
            used.add(key)
            out.append({
                "base": valid[int(pos)], "donor": valid[donor],
                "axis": "horizontal" if axis == 0 else "longitudinal",
                "kind": "large_position", "state_distance": float(np.min(dist)),
                "target_delta": float(values[donor] - values[pos]),
            })
            if maximum is not None and len(out) >= maximum:
                break
        return out

    # The contract asks for >=20 pairs.  Use a fixed deterministic cap once
    # that target is met; running thousands of redundant pairs is not a new
    # scientific condition and would dominate the diagnostic cost.
    horizontal = matched_pairs(0, max(0.5 * x_span, 1e-6), maximum=20)
    longitudinal = matched_pairs(1, max(0.5 * y_span, 1e-6), maximum=20)

    def small_position_pairs(maximum: int = 20) -> list[dict]:
        out: list[dict] = []
        used: set[tuple[int, int]] = set()
        for i in rng.permutation(len(valid)):
            candidates = np.arange(len(valid))
            candidates = candidates[candidates != i]
            phase_ok = np.abs(times[candidates] - times[i]) <= 0.20
            candidates = candidates[phase_ok]
            if len(candidates) == 0:
                continue
            pos_dist = np.sqrt((xs[candidates] - xs[i]) ** 2 + (ys[candidates] - ys[i]) ** 2)
            # Same-position control is defined relative to the observed spans.
            candidates = candidates[pos_dist <= 0.10 * max(math.hypot(x_span, y_span), 1e-6)]
            if len(candidates) == 0:
                continue
            state_dist = np.linalg.norm(z[candidates] - z[i], axis=1)
            donor = int(candidates[np.argmin(state_dist)])
            key = (int(i), donor)
            if key in used:
                continue
            used.add(key)
            out.append({"base": valid[int(i)], "donor": valid[donor], "axis": "both",
                        "kind": "same_position", "state_distance": float(np.min(state_dist)),
                        "target_delta_x": float(xs[donor] - xs[i]),
                        "target_delta_y": float(ys[donor] - ys[i])})
            if len(out) >= maximum:
                break
        return out

    same = small_position_pairs()
    unmatched = []
    perm = rng.permutation(len(valid))
    for i, j in zip(range(len(valid)), perm, strict=True):
        if i == int(j):
            continue
        unmatched.append({"base": valid[i], "donor": valid[int(j)], "axis": "both",
                          "kind": "random_unmatched", "state_distance": None,
                          "target_delta_x": float(xs[int(j)] - xs[i]),
                          "target_delta_y": float(ys[int(j)] - ys[i])})
        if len(unmatched) >= 20:
            break
    return ({"horizontal": horizontal, "longitudinal": longitudinal, "same_position": same,
             "random_unmatched": unmatched},
            {"eligible_frames": len(valid), "x_span": x_span, "y_span": y_span,
             "state_scale": scale.tolist(), "confidence_min": CONFIDENCE_MIN,
             "phase_max_fraction": PHASE_MAX_FRACTION, "state_time_tolerance": 0.20,
             "large_delta_fraction": 0.50, "same_position_fraction": 0.10,
             "selection_seed": seed})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--episodes", default=",".join(map(str, DEV_EPISODES)))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--allow-nonp1-checkpoint", action="store_true",
                    help="Reuse the frozen pairing diagnostic for a new Discovery checkpoint.")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    policy = ACTPolicy.from_pretrained(ckpt, config=cfg)
    policy.eval()
    device = next(policy.parameters()).device
    pre, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(ckpt))
    episodes = parse_episodes(args.episodes)
    if episodes != DEV_EPISODES:
        raise ValueError(f"T0.2 is frozen to Dev episodes {DEV_EPISODES}, got {episodes}")
    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=episodes, video_backend="pyav")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    collate_fn=collate, drop_last=False)
    oc = policy.model.object_conditioning
    top_idx = policy.model.object_conditioning_top_camera_index

    # Pass 1: collect frozen eligibility metadata only.
    # Episode lengths come from metadata; do not decode every video frame just
    # to infer timing (that would make the diagnostic needlessly expensive).
    lengths = {int(row["episode_index"]): int(row["length"])
               for row in ds.meta.episodes.to_list() if int(row["episode_index"]) in episodes}
    records: list[dict] = []
    dataset_cursor = 0
    with torch.no_grad():
        for batch in dl:
            original_ep = batch["episode_index"].detach().cpu().numpy().reshape(-1)
            original_frame = batch["frame_index"].detach().cpu().numpy().reshape(-1)
            original_index = batch["index"].detach().cpu().numpy().reshape(-1)
            original_state = batch["observation.state"].detach().cpu().numpy()
            batch = move_batch(pre(batch), device)
            if cfg.image_features:
                batch[OBS_IMAGES] = [batch[k] for k in cfg.image_features]
            raw, sim = oc.extract_raw_object_tokens(batch[OBS_IMAGES][top_idx])
            coords = raw[:, 0, 384:386].detach().cpu().numpy()
            conf = sim.detach().cpu().numpy()
            if conf.ndim > 1:
                conf = conf.reshape(conf.shape[0], -1).max(axis=1)
            for i in range(len(original_ep)):
                ep = int(original_ep[i]); frame = int(original_frame[i])
                frac = frame / max(lengths[ep] - 1, 1)
                finite = bool(np.isfinite(coords[i]).all() and np.isfinite(conf[i]))
                records.append({"episode": ep, "frame": frame, "index": int(original_index[i]),
                                "dataset_index": int(dataset_cursor + i),
                                "state": original_state[i].astype(float).tolist(),
                                "x": float(coords[i, 0]), "y": float(coords[i, 1]),
                                "confidence": float(conf[i]), "time_fraction": float(frac),
                                "eligible": bool(finite and conf[i] >= CONFIDENCE_MIN and frac <= PHASE_MAX_FRACTION)})
            dataset_cursor += len(original_ep)

    pairs, selection = build_pair_candidates(records, args.seed)
    manifest = {"checkpoint": str(ckpt), "episodes": episodes, "records": records,
                "selection": selection, "pairs": pairs,
                "frozen_primary_dims": {"horizontal": X_DIM, "longitudinal": Y_DIM},
                "expert_correlations": {"x_dim0": EXPERT_X_CORR, "y_dim2": EXPERT_Y_CORR}}
    (out / "pair_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def inject(src_batch: dict, tokens: torch.Tensor) -> torch.Tensor:
        original = oc.forward
        try:
            def forward_with_tokens(act_tokens, top_image, object_tokens=None):
                kv = tokens.transpose(0, 1)
                response, _ = oc.cross_attention(act_tokens, kv, kv, need_weights=False)
                return act_tokens + oc.gate * response
            oc.forward = forward_with_tokens
            return policy.model(src_batch)[0]
        finally:
            oc.forward = original

    variants = ("whole", "local_only", "global_only")
    results: list[dict] = []
    # Run only the requested large/same/random pair manifests.  Each source is
    # evaluated with the identical base observation; only object tokens change.
    all_pairs = []
    for group in ("horizontal", "longitudinal", "same_position", "random_unmatched"):
        for pair in pairs[group]:
            item = dict(pair); item["group"] = group; all_pairs.append(item)
    for start in range(0, len(all_pairs), args.batch_size):
        chunk = all_pairs[start:start + args.batch_size]
        src_items = [ds[int(p["base"]["dataset_index"])] for p in chunk]
        donor_items = [ds[int(p["donor"]["dataset_index"])] for p in chunk]
        src_batch = move_batch(pre(collate(src_items)), device)
        donor_batch = move_batch(pre(collate(donor_items)), device)
        if cfg.image_features:
            src_batch[OBS_IMAGES] = [src_batch[k] for k in cfg.image_features]
            donor_batch[OBS_IMAGES] = [donor_batch[k] for k in cfg.image_features]
        with torch.no_grad():
            normal = action_h0(policy.model(src_batch)[0])
            donor_raw, _ = oc.extract_raw_object_tokens(donor_batch[OBS_IMAGES][top_idx])
            variant_preds = {}
            for variant in variants:
                raw = donor_raw
                if variant == "local_only":
                    base_raw, _ = oc.extract_raw_object_tokens(src_batch[OBS_IMAGES][top_idx])
                    raw = base_raw.clone(); raw[:, 1:] = donor_raw[:, 1:]
                elif variant == "global_only":
                    base_raw, _ = oc.extract_raw_object_tokens(src_batch[OBS_IMAGES][top_idx])
                    raw = base_raw.clone(); raw[:, 0] = donor_raw[:, 0]
                variant_preds[variant] = action_h0(inject(src_batch, oc.adapt_object_tokens(raw)))
        for i, pair in enumerate(chunk):
            row = {"group": pair["group"], "base": pair["base"], "donor": pair["donor"],
                   "target_delta_x": float(pair.get("target_delta_x", pair["donor"]["x"] - pair["base"]["x"])),
                   "target_delta_y": float(pair.get("target_delta_y", pair["donor"]["y"] - pair["base"]["y"])),
                   "variants": {}}
            for variant in variants:
                delta = (variant_preds[variant][i] - normal[i]).detach().cpu().numpy()
                row["variants"][variant] = {"action_delta_h0": delta.tolist(),
                                             "arm_l1": float(np.mean(np.abs(delta[:-1]))),
                                             "gripper_abs": float(abs(delta[-1])),
                                             "all_l1": float(np.mean(np.abs(delta)))}
            results.append(row)

    (out / "transplant_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    def summarise(rows: list[dict], variant: str, axis: str) -> dict:
        selected = [r for r in rows if r["group"] == axis]
        dx = np.asarray([r["target_delta_x"] for r in selected], float)
        dy = np.asarray([r["target_delta_y"] for r in selected], float)
        da = np.asarray([r["variants"][variant]["action_delta_h0"] for r in selected], float)
        def direction(target, action, corr):
            expected = np.sign(target) * np.sign(corr)
            observed = np.sign(action)
            mask = (np.abs(target) > SIGN_EPS) & (np.abs(action) > SIGN_EPS)
            return {"rate": float(np.mean(observed[mask] == expected[mask])) if mask.any() else 0.0,
                    "n_nonzero": int(mask.sum()), "n_total": int(len(target))}
        return {"n": len(selected), "mean_abs_h0": float(np.mean(np.abs(da))) if len(da) else 0.0,
                "mean_abs_arm": float(np.mean(np.abs(da[:, :-1]))) if len(da) else 0.0,
                "mean_abs_gripper": float(np.mean(np.abs(da[:, -1]))) if len(da) else 0.0,
                "pearson_dx_dim0": pearson(dx, da[:, X_DIM]) if len(da) else 0.0,
                "spearman_dx_dim0": spearman(dx, da[:, X_DIM]) if len(da) else 0.0,
                "pearson_dy_dim2": pearson(dy, da[:, Y_DIM]) if len(da) else 0.0,
                "spearman_dy_dim2": spearman(dy, da[:, Y_DIM]) if len(da) else 0.0,
                "horizontal_direction": direction(dx, da[:, X_DIM], EXPERT_X_CORR) if len(da) else {},
                "longitudinal_direction": direction(dy, da[:, Y_DIM], EXPERT_Y_CORR) if len(da) else {},
                "axis_selectivity": {"horizontal": float(np.mean(np.abs(da[:, X_DIM])) / max(np.mean(np.abs(da[:, :-1])), 1e-12)) if len(da) else 0.0,
                                     "longitudinal": float(np.mean(np.abs(da[:, Y_DIM])) / max(np.mean(np.abs(da[:, :-1])), 1e-12)) if len(da) else 0.0}}

    metrics = {variant: {group: summarise(results, variant, group)
                         for group in ("horizontal", "longitudinal", "same_position", "random_unmatched")}
               for variant in variants}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    controls = {variant: {group: summarise(results, variant, group)
                          for group in ("same_position", "random_unmatched")}
                for variant in variants}
    (out / "controls.json").write_text(json.dumps(controls, indent=2), encoding="utf-8")
    validation = {"checkpoint_frozen": (args.allow_nonp1_checkpoint or ("P1_ocfix2_s1" in str(ckpt) and "100000" in str(ckpt))),
                  "episodes_frozen": episodes == DEV_EPISODES, "final_test_used": False,
                  "training_run": False, "real_robot_used": False,
                  "token_variants": list(variants), "pair_counts": {k: len(v) for k, v in pairs.items()},
                  "confidence_min": CONFIDENCE_MIN, "phase_rule": f"time_fraction <= {PHASE_MAX_FRACTION}; no phase labels available"}
    (out / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps({"pair_counts": validation["pair_counts"], "device": str(device), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
