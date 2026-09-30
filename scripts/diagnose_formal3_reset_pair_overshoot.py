"""Diagnose reset-pair 10k magnitude overshoot from existing eval artifacts.

No new data, no training: reads the 10k dev12/cross npz files, the original
dataset parquet and the time-trim manifest, and answers:
1. Is the ~2.9x h0 overshoot specific to the paired reset observations
   (rows 0..14) or present across the task interval too?
2. Which episode properties correlate with the overshoot (demonstrated
   first-motion size, trim start S, protocol group ep<=37 vs ep>=38)?
3. What do the direction-failing episodes (43, 55) share?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

EPISODES = (9, 11, 17, 25, 26, 28, 30, 31, 43, 52, 55, 59)
ARM = 5


def load_npz(directory: Path, episode: int) -> dict[str, np.ndarray]:
    with np.load(directory / f"episode_{episode:03d}_chunks.npz") as data:
        return {key: data[key] for key in data.files}


def load_original(original_root: Path) -> dict[int, dict[str, np.ndarray]]:
    table = pq.read_table(original_root / "data/chunk-000/file-000.parquet",
                          columns=["episode_index", "frame_index", "action", "observation.state"])
    out = {}
    for ep in EPISODES:
        rows = table.filter(pc.equal(table["episode_index"], ep)).sort_by("frame_index")
        out[ep] = {
            "action": np.stack(rows["action"].to_numpy()),
            "state": np.stack(rows["observation.state"].to_numpy()),
        }
    return out


def row_ratio(pred_row: np.ndarray, truth_row: np.ndarray, baseline: np.ndarray) -> float:
    desired = truth_row[:ARM] - baseline[:ARM]
    pred = pred_row[:ARM] - baseline[:ARM]
    desired_norm = float(np.linalg.norm(desired))
    return float(np.linalg.norm(pred) / desired_norm) if desired_norm else np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--timetrim-campaign", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--timetrim-root", type=Path, required=True)
    args = parser.parse_args()

    dev12_root = args.campaign / "eval_dev12" / "RESETPAIR_010000"
    cross_root = args.campaign / "eval_cross_on_timetrim" / "RESETPAIR_ON_TIMETRIM_010000"
    tt_cross_root = args.timetrim_campaign / "eval_initial_cross" / "TIMETRIM_ON_ORIGINAL_010000"
    boundaries = {row["episode_index"]: row
                  for row in json.loads((args.timetrim_root / "meta/time_trim_manifest.json").read_text())["boundaries"]}
    original = load_original(args.original_root)

    per_episode = {}
    for ep in EPISODES:
        S = boundaries[ep]["start"]
        actions = original[ep]["action"]
        baseline = np.median(actions[0:15], axis=0)
        desired = actions[S]
        data = load_npz(dev12_root, ep)
        pred, truth, state = data["prediction_raw"], data["truth_raw"], data["state"]
        tt_cross = load_npz(tt_cross_root, ep)
        tt_pred_h0 = tt_cross["prediction_raw"][0, 0]

        # h0 metrics at the reset observation (row 0 of the reset-pair episode).
        pred_delta = pred[0, 0, :ARM] - baseline[:ARM]
        desired_delta = desired[:ARM] - baseline[:ARM]
        desired_norm = float(np.linalg.norm(desired_delta))
        pred_norm = float(np.linalg.norm(pred_delta))
        cosine = float(np.dot(desired_delta, pred_delta) / (desired_norm * pred_norm)) if desired_norm * pred_norm else np.nan
        tt_delta = tt_pred_h0[:ARM] - baseline[:ARM]
        tt_norm = float(np.linalg.norm(tt_delta))
        tt_ratio = tt_norm / desired_norm if desired_norm else np.nan

        # Which truth row of the first 30 chunk rows is the predicted h0 closest to?
        closest = int(np.argmin(np.linalg.norm(truth[0, :30, :ARM] - pred[0, 0, :ARM], axis=1)))

        # Overshoot contrast: reset-observation rows (0..14) vs task rows (15..30).
        reset_rows = [row_ratio(pred[r, 0], truth[r, 0], baseline) for r in range(0, 15)]
        task_rows = [row_ratio(pred[r, 0], truth[r, 0], baseline) for r in range(15, 31)]
        tt_rows = [row_ratio(tt_cross["prediction_raw"][0, r], tt_cross["truth_raw"][0, r], baseline) for r in range(0, 15)]

        per_episode[str(ep)] = {
            "trim_start_S": S,
            "desired_first_motion_norm": desired_norm,
            "pred_h0_norm": pred_norm,
            "vector_ratio": pred_norm / desired_norm if desired_norm else np.nan,
            "cosine": cosine,
            "pan_sign_matches": bool(np.sign(pred_delta[0]) == np.sign(desired_delta[0]) and np.sign(desired_delta[0]) != 0),
            "per_joint_ratio": [float(pred_delta[j] / desired_delta[j]) if abs(desired_delta[j]) > 0.05 else np.nan
                                for j in range(ARM)],
            "per_joint_desired_delta": desired_delta.tolist(),
            "closest_truth_row_in_first30": closest,
            "reset_rows_ratio_mean": float(np.nanmean(reset_rows)),
            "task_rows_ratio_mean": float(np.nanmean(task_rows)),
            "timetrim_ratio": tt_ratio,
            "timetrim_reset_rows_ratio_mean": float(np.nanmean(tt_rows)),
            "protocol_group": "ep0-37" if ep <= 37 else "ep38-59",
            "reset_state_deg": state[0].tolist(),
        }

    # Aggregates.
    ratios = np.asarray([v["vector_ratio"] for v in per_episode.values()])
    desired_norms = np.asarray([v["desired_first_motion_norm"] for v in per_episode.values()])
    starts = np.asarray([v["trim_start_S"] for v in per_episode.values()])
    tt_ratios = np.asarray([v["timetrim_ratio"] for v in per_episode.values()])
    reset_means = np.asarray([v["reset_rows_ratio_mean"] for v in per_episode.values()])
    task_means = np.asarray([v["task_rows_ratio_mean"] for v in per_episode.values()])

    def corr(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 3:
            return np.nan
        a, b = a[mask] - a[mask].mean(), b[mask] - b[mask].mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / denom) if denom else np.nan

    protocol_groups = {}
    keys = list(per_episode.keys())
    for group in ("ep0-37", "ep38-59"):
        mask = np.asarray([per_episode[k]["protocol_group"] == group for k in keys])
        protocol_groups[group] = {
            "episodes": [k for k, m in zip(keys, mask) if m],
            "vector_ratio_mean": float(np.nanmean(ratios[mask])) if mask.any() else np.nan,
        }

    report = {
        "per_episode": per_episode,
        "aggregates": {
            "vector_ratio_median": float(np.median(ratios)),
            "corr_ratio_vs_desired_norm": corr(ratios, desired_norms),
            "corr_ratio_vs_trim_start": corr(ratios, starts),
            "corr_ratio_vs_timetrim_ratio": corr(ratios, tt_ratios),
            "reset_rows_ratio_mean": float(np.mean(reset_means)),
            "task_rows_ratio_mean": float(np.mean(task_means)),
            "overshoot_episodes_ratio_gt_2_5": [k for k, v in per_episode.items() if v["vector_ratio"] > 2.5],
            "direction_fail_episodes": [k for k, v in per_episode.items() if v["cosine"] < 0],
            "protocol_groups": protocol_groups,
        },
    }
    out_json = args.campaign / "diagnosis_overshoot.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# Reset-pair 10k overshoot diagnosis (existing artifacts, no retraining)", "",
             "| ep | S | first-motion norm | ratio | cosine | closest truth row | reset rows mean | task rows mean | tt ratio | group |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for ep in EPISODES:
        v = per_episode[str(ep)]
        lines.append(f"| {ep} | {v['trim_start_S']} | {v['desired_first_motion_norm']:.2f} | {v['vector_ratio']:.2f} | "
                     f"{v['cosine']:.2f} | {v['closest_truth_row_in_first30']} | {v['reset_rows_ratio_mean']:.2f} | "
                     f"{v['task_rows_ratio_mean']:.2f} | {v['timetrim_ratio']:.2f} | {v['protocol_group']} |")
    agg = report["aggregates"]
    lines += ["",
              f"- ratio median: {agg['vector_ratio_median']:.3f}",
              f"- corr(ratio, first-motion norm): {agg['corr_ratio_vs_desired_norm']:.3f}",
              f"- corr(ratio, trim start S): {agg['corr_ratio_vs_trim_start']:.3f}",
              f"- corr(ratio, timetrim ratio same ep): {agg['corr_ratio_vs_timetrim_ratio']:.3f}",
              f"- reset-obs rows (0-14) ratio mean: {agg['reset_rows_ratio_mean']:.2f}; task rows (15-30) ratio mean: {agg['task_rows_ratio_mean']:.2f}",
              f"- overshoot (ratio>2.5): {agg['overshoot_episodes_ratio_gt_2_5']}",
              f"- direction fail (cosine<0): {agg['direction_fail_episodes']}",
              f"- protocol groups: {json.dumps(agg['protocol_groups'], ensure_ascii=False)}"]
    (args.campaign / "diagnosis_overshoot.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_json)


if __name__ == "__main__":
    main()
