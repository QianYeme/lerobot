#!/usr/bin/env python
"""G0 fixed-Dev position-generalization screening; no training or Final Test."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy  # registers ACT configs
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.offline_eval_act_det import collate, parse_episodes
from lerobot.utils.constants import ACTION, OBS_IMAGES


DEV = [3, 5, 12, 15, 21, 28, 33, 39, 41, 47, 53, 58]


def move(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def action_slice(pred, n):
    return pred[:, :n] if pred.ndim == 3 else pred[:, None, :]


def valid_error(pred, target, pad, n=None):
    if n is not None:
        pred, target, pad = action_slice(pred, n), action_slice(target, n), pad[:, :n]
    mask = (~pad).unsqueeze(-1)
    err = (pred - target).abs() * mask
    denom = mask.expand_as(err).sum(dim=(1, 2)).clamp_min(1)
    return err.sum(dim=(1, 2)) / denom, err


def l1_group(values, groups, key):
    ix = np.flatnonzero(groups == key)
    return float(np.mean(values[ix])) if len(ix) else None


def summary_distribution(preds, groups):
    # Predicted action differences are reported for first 1/5/10/full available steps.
    out = {}
    for horizon in [1, 5, 10, preds.shape[1]]:
        n = min(horizon, preds.shape[1]); z = preds[:, :n].mean(axis=1)
        means = {}; within = {}
        for g in ["left", "center", "right", "near", "far", "edge", "non_edge"]:
            ix = np.flatnonzero(groups == g)
            if len(ix):
                means[g] = z[ix].mean(axis=0).tolist()
                within[g] = float(np.mean(np.sum((z[ix] - z[ix].mean(0)) ** 2, axis=1)))
        distances = {}
        for a, b in [("left", "right"), ("near", "far"), ("edge", "non_edge")]:
            if a in means and b in means:
                distances[f"{a}_vs_{b}_l2"] = float(np.linalg.norm(np.asarray(means[a]) - np.asarray(means[b])))
        out[str(horizon)] = {"means": means, "within_group_variance": within, "between_group_distance": distances}
    return out


def extract_positions(ds, checkpoint, batch_size, workers, device):
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    policy = ACTPolicy.from_pretrained(checkpoint, config=cfg).to(device).eval()
    pre, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(checkpoint))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate)
    pos = {}
    oc = getattr(policy.model, "object_conditioning", None)
    if oc is None:
        raise RuntimeError("Frozen P0.1 position extraction requires an object-conditioning checkpoint")
    top_idx = policy.model.object_conditioning_top_camera_index
    with torch.no_grad():
        for raw in dl:
            ids = raw["index"].numpy().reshape(-1)
            batch = move(pre(raw), device)
            batch[OBS_IMAGES] = [batch[k] for k in cfg.image_features]
            rawtok, sim = oc.extract_raw_object_tokens(batch[OBS_IMAGES][top_idx])
            spatial = oc.global_spatial_features(rawtok).detach().cpu().numpy()
            conf = sim.detach().cpu().numpy().reshape(-1)
            for i, idx in enumerate(ids):
                pos[int(idx)] = [float(spatial[i, 0]), float(spatial[i, 1]), float(spatial[i, 2]), float(conf[i])]
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True); ap.add_argument("--root", required=True)
    ap.add_argument("--position-checkpoint", required=True)
    ap.add_argument("--candidate", action="append", required=True, help="LABEL=checkpoint")
    ap.add_argument("--output-dir", required=True); ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=DEV, video_backend="pyav")
    pos = extract_positions(ds, Path(args.position_checkpoint), args.batch_size, args.num_workers, device)
    np.savez(out / "dev_positions.npz", ids=np.asarray(list(pos)), values=np.asarray(list(pos.values())))
    vals = np.asarray(list(pos.values()), float); x, y = vals[:, 0], vals[:, 1]
    xq, yq = np.quantile(x, [1/3, 2/3]), np.quantile(y, [1/3, 2/3]); xm, ym = np.median(x), np.median(y)
    edge_score = np.maximum(np.abs(x - xm), np.abs(y - ym)); edge_cut = float(np.quantile(edge_score, .75))
    position_meta = {"x_tertiles": xq.tolist(), "y_tertiles": yq.tolist(), "median_x": float(xm), "median_y": float(ym), "edge_cut_top25_abs_median_distance": edge_cut, "episodes": DEV, "final_test_used": False}
    (out / "position_groups.json").write_text(json.dumps(position_meta, indent=2))
    def group_for(idx):
        xx, yy = pos[int(idx)][:2]; score = max(abs(xx - xm), abs(yy - ym))
        return ["left", "center", "right"][0 if xx < xq[0] else 2 if xx >= xq[1] else 1], ["near", "center", "far"][0 if yy < yq[0] else 2 if yy >= yq[1] else 1], "edge" if score >= edge_cut else "non_edge"
    groups = {}
    for idx in pos:
        h, v, e = group_for(idx); groups[idx] = (h, v, e)
    manifest = {"position_groups": position_meta, "sample_groups": {str(k): groups[k] for k in groups}, "fixed_before_model_comparison": True}
    (out / "position_group_manifest.json").write_text(json.dumps(manifest, indent=2))

    results = {}
    for spec in args.candidate:
        label, ckpt = spec.split("=", 1); ckpt = Path(ckpt)
        cfg = PreTrainedConfig.from_pretrained(ckpt); policy = ACTPolicy.from_pretrained(ckpt, config=cfg).to(device).eval()
        pre, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(ckpt))
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
        rows = []; chunk_preds = []; chunk_groups = []; phase_reach = set()
        with torch.no_grad():
            for raw in dl:
                ids = raw["index"].numpy().reshape(-1); batch = pre(raw); batch = move(batch, device); batch[OBS_IMAGES] = [batch[k] for k in cfg.image_features]
                pred = policy.model(batch)[0]; target = batch[ACTION]
                # Formal4 samples contain a single teacher-forced target action [B,6],
                # while ACT predicts an action chunk [B,T,6].  Only h0 has a direct
                # target; later/full-chunk values are prediction-distribution audits.
                if target.ndim == 2:
                    target = target[:, None, :]
                pad = batch.get("action_is_pad", torch.zeros(target.shape[:-1], dtype=torch.bool, device=device))
                if pad.ndim == 1: pad = pad[:, None]
                pred_np, target_np, pad_np = pred.detach().cpu().numpy(), target.detach().cpu().numpy(), pad.detach().cpu().numpy()
                chunk_preds.append(pred_np); chunk_groups.extend([groups[int(i)] for i in ids])
                for j, idx in enumerate(ids):
                    g = groups[int(idx)]; per = {}
                    for n in [1, 5, 10, pred.shape[1]]:
                        n = min(n, pred.shape[1])
                        if target_np.shape[1] < n:
                            per[str(n)] = {"all": None, "arm": None, "gripper": None, "note": "dataset exposes only h0 target; use prediction_distribution for chunk"}
                        else:
                            valid = ~pad_np[j, :n]; ee = np.abs(pred_np[j, :n] - target_np[j, :n])[valid]
                            per[str(n)] = {"all": float(ee.mean()) if ee.size else None, "arm": float(ee[:, :-1].mean()) if ee.size else None, "gripper": float(ee[:, -1].mean()) if ee.size else None}
                    rows.append({"index": int(idx), "horizontal": g[0], "vertical": g[1], "edge": g[2], "metrics": per})
        preds = np.concatenate(chunk_preds, axis=0); garr = np.asarray([f"{a}|{b}|{c}" for a,b,c in chunk_groups])
        # A separate categorical array for distribution calculations.
        dist = {}
        for key, posn in [("horizontal",0),("vertical",1),("edge",2)]: dist[key] = np.asarray([q[posn] for q in chunk_groups])
        strat = defaultdict(lambda: {"n": 0, "all": [], "arm": [], "gripper": []})
        for r in rows:
            for axis in ["horizontal", "vertical", "edge"]:
                key = f"{axis}:{r[axis]}"; z = r["metrics"][str(min(1, preds.shape[1]))]; strat[key]["n"] += 1; strat[key]["all"].append(z["all"]); strat[key]["arm"].append(z["arm"]); strat[key]["gripper"].append(z["gripper"])
        strat_out = {k: {"n": v["n"], "all_l1": float(np.mean(v["all"])), "arm_l1": float(np.mean(v["arm"])), "gripper_l1": float(np.mean(v["gripper"]))} for k,v in strat.items()}
        results[label] = {"checkpoint": str(ckpt), "frames": len(rows), "overall": strat_out, "prediction_distribution": {k: summary_distribution(preds, dist[k]) for k in dist}, "sample_rows": rows}
        (out / f"{label}.json").write_text(json.dumps(results[label], indent=2))
    (out / "stratified_metrics.json").write_text(json.dumps(results, indent=2))
    (out / "validation.json").write_text(json.dumps({"experiment":"G0_POSITION_GENERALIZATION_SCREENING","dev_episodes":DEV,"final_test_used":False,"policy_retraining":False,"architecture_change":False,"real_robot_used":False,"device":str(device),"status":"complete"}, indent=2))
    print(json.dumps({"models": list(results), "frames": {k:v["frames"] for k,v in results.items()}, "device":str(device)}, indent=2))


if __name__ == "__main__": main()
