"""Summarize the preregistered Formal3 time-trim 10k comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


EPISODES = (9, 11, 17, 25, 26, 28, 30, 31, 43, 52, 55, 59)
STEPS = (2000, 4000, 6000, 8000, 10000)
PHASES = ("approach", "grasp", "lift_place", "release")


def departure(chunk: np.ndarray, baseline: np.ndarray, threshold: float) -> int | None:
    moved = np.max(np.abs(chunk[:, :5] - baseline[None, :5]), axis=1) >= threshold
    indices = np.flatnonzero(moved)
    return int(indices[0]) if len(indices) else None


def initial_metrics(chunks: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]], threshold: float) -> dict:
    departures, gt_departures, direction_ok = [], [], []
    first_target_arm_mae, motion_cosine, motion_norm_ratio = [], [], []
    leave = {n: [] for n in (1, 5, 10)}
    per_episode = {}
    for ep, (prediction, truth, _state) in chunks.items():
        baseline = np.median(truth[:15], axis=0)
        pred_h = departure(prediction, baseline, threshold)
        gt_h = departure(truth, baseline, threshold)
        departures.append(np.nan if pred_h is None else pred_h)
        gt_departures.append(np.nan if gt_h is None else gt_h)
        for n in leave:
            leave[n].append(pred_h is not None and pred_h < n)
        correct = None
        if pred_h is not None and gt_h is not None:
            pred_delta = prediction[pred_h, 0] - baseline[0]
            gt_delta = truth[gt_h, 0] - baseline[0]
            correct = bool(np.sign(pred_delta) == np.sign(gt_delta) and np.sign(gt_delta) != 0)
            direction_ok.append(correct)
            desired = truth[gt_h, :5]
            predicted_h0 = prediction[0, :5]
            first_target_arm_mae.append(float(np.abs(predicted_h0 - desired).mean()))
            desired_delta = desired - baseline[:5]
            predicted_delta = predicted_h0 - baseline[:5]
            denominator = float(np.linalg.norm(desired_delta) * np.linalg.norm(predicted_delta))
            motion_cosine.append(float(np.dot(desired_delta, predicted_delta) / denominator) if denominator else np.nan)
            desired_norm = float(np.linalg.norm(desired_delta))
            motion_norm_ratio.append(float(np.linalg.norm(predicted_delta) / desired_norm) if desired_norm else np.nan)
        per_episode[str(ep)] = {"departure_h": pred_h, "gt_departure_h": gt_h, "pan_direction_ok": correct}
    finite = np.asarray(departures, dtype=float)
    finite = finite[np.isfinite(finite)]
    gt_finite = np.asarray(gt_departures, dtype=float)
    gt_finite = gt_finite[np.isfinite(gt_finite)]
    return {
        "departure_h": {
            "median": float(np.median(finite)) if len(finite) else None,
            "min": int(finite.min()) if len(finite) else None,
            "max": int(finite.max()) if len(finite) else None,
        },
        "gt_departure_h_median": float(np.median(gt_finite)) if len(gt_finite) else None,
        "leave_rate": {f"n{n}": float(np.mean(values)) for n, values in leave.items()},
        "pan_direction_rate": float(np.mean(direction_ok)) if direction_ok else None,
        "h0_to_first_gt_motion_arm_mae": float(np.mean(first_target_arm_mae)) if first_target_arm_mae else None,
        "h0_motion_cosine_mean": float(np.nanmean(motion_cosine)) if motion_cosine else None,
        "h0_motion_cosine_positive_rate": float(np.mean(np.asarray(motion_cosine) > 0)) if motion_cosine else None,
        "h0_motion_norm_ratio_median": float(np.nanmedian(motion_norm_ratio)) if motion_norm_ratio else None,
        "episodes": per_episode,
    }


def mean_error(errors: list[np.ndarray]) -> dict:
    values = np.concatenate(errors, axis=0)
    joint = values.mean(axis=0)
    return {"arm": float(joint[:5].mean()), "gripper": float(joint[5]), "per_joint": joint.tolist()}


def load_npz(directory: Path, episode: int) -> dict[str, np.ndarray]:
    with np.load(directory / f"episode_{episode:03d}_chunks.npz") as data:
        return {key: data[key] for key in data.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--trimmed-root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=5.0)
    args = parser.parse_args()
    eval_root = args.campaign / "eval_dev12"
    cross_root = args.campaign / "eval_initial_cross"
    if not (eval_root / "evaluation.done").is_file() or not (cross_root / "evaluation.done").is_file():
        raise FileNotFoundError("Evaluation families are not complete")

    manifest = json.loads((args.trimmed_root / "meta/time_trim_manifest.json").read_text())
    boundaries = {row["episode_index"]: row for row in manifest["boundaries"]}
    phase_table = pq.read_table(args.trimmed_root / "meta/reviewed_phase_labels.parquet")
    phase_by_episode = {}
    for ep in EPISODES:
        mask = np.asarray(phase_table["episode_index"].to_numpy()) == ep
        phase_by_episode[ep] = np.asarray(phase_table["phase"].to_pylist(), dtype=object)[mask]

    report = {"threshold_degrees": args.threshold, "episodes": list(EPISODES), "checkpoints": {}}
    for step in STEPS:
        tag = f"{step:06d}"
        original_dir = eval_root / f"ORIGINAL_{tag}"
        trimmed_dir = eval_root / f"TIMETRIM_{tag}"
        cross_dir = cross_root / f"TIMETRIM_ON_ORIGINAL_{tag}"
        original_initial, trimmed_initial = {}, {}
        common_errors = {model: {phase: [] for phase in PHASES} for model in ("original", "timetrim")}
        common_all = {"original": [], "timetrim": []}
        for ep in EPISODES:
            original = load_npz(original_dir, ep)
            trimmed = load_npz(trimmed_dir, ep)
            cross = load_npz(cross_dir, ep)
            original_initial[ep] = (original["prediction_raw"][0], original["truth_raw"][0], original["state"][0])
            trimmed_initial[ep] = (cross["prediction_raw"][0], cross["truth_raw"][0], cross["state"][0])
            boundary = boundaries[ep]
            original_error = np.abs(
                original["prediction_raw"][boundary["start"] : boundary["end"], 0]
                - original["truth_raw"][boundary["start"] : boundary["end"], 0]
            )
            trimmed_error = np.abs(trimmed["prediction_raw"][:, 0] - trimmed["truth_raw"][:, 0])
            if original_error.shape != trimmed_error.shape:
                raise ValueError(f"Common interval length mismatch for episode {ep}")
            common_all["original"].append(original_error)
            common_all["timetrim"].append(trimmed_error)
            phases = phase_by_episode[ep]
            for phase in PHASES:
                mask = phases == phase
                if mask.any():
                    common_errors["original"][phase].append(original_error[mask])
                    common_errors["timetrim"][phase].append(trimmed_error[mask])
        checkpoint = {
            "initial_same_original_observation": {
                "original": initial_metrics(original_initial, args.threshold),
                "timetrim": initial_metrics(trimmed_initial, args.threshold),
            },
            "common_trimmed_interval_h0_mae": {
                model: {
                    "all": mean_error(common_all[model]),
                    "phases": {phase: mean_error(values) for phase, values in common_errors[model].items()},
                }
                for model in ("original", "timetrim")
            },
        }
        report["checkpoints"][str(step)] = checkpoint

    out_json = args.campaign / "summary.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Formal3 time-trim P3 evaluation",
        "",
        "Same original deployment-frame-0 observations; departure is an arm target at least 5 degrees from the median original target in h0..h14.",
        "",
        "| step | model | departure h median/min/max | leave n=1/5/10 | pan direction | common arm MAE | gripper MAE |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    for step in STEPS:
        item = report["checkpoints"][str(step)]
        for model in ("original", "timetrim"):
            initial = item["initial_same_original_observation"][model]
            dep = initial["departure_h"]
            leave = initial["leave_rate"]
            mae = item["common_trimmed_interval_h0_mae"][model]["all"]
            lines.append(
                f"| {step} | {model} | {dep['median']}/{dep['min']}/{dep['max']} | "
                f"{leave['n1']:.2f}/{leave['n5']:.2f}/{leave['n10']:.2f} | "
                f"{initial['pan_direction_rate'] if initial['pan_direction_rate'] is not None else 'NA'} | "
                f"{mae['arm']:.4f} | {mae['gripper']:.4f} |"
            )
    (args.campaign / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_json)


if __name__ == "__main__":
    main()
