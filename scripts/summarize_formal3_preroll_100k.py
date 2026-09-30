"""Summarize the preregistered Formal3 preroll 100k (B1) evaluation.

Gate set (preregistered 2026-09-29 in docc/方案与规划/Formal3_起步方向修复_完整规划_2026-09-29.md),
applied at the final checkpoint (default 100000):
- G1: train48 frame0 h0 direction cosine+ >= 33/48
- G2: dev12 frame0 h0 direction cosine+ >= 9/12
- G3: dev12 frame0 h0 motion norm ratio median in [0.5, 2.0]
- G4: train48 frame0 h0 |bias| (arm mean abs, degrees) <= 2.5
- G5: task-interval MAE (arm/gripper/phases) vs frozen time-trim 10k <= 1.2
- G6: data/training/eval integrity

Metric conventions are identical to the frozen 10k summarizer
(summarize_formal3_reset_pair_p3.py, left untouched as evidence) and
analyze_formal3_h0_bias.py: baseline = median of the original episode's first
15 actions, desired = original action at trim start S, first 5 action dims =
arm, values natively degrees. The time-trim reference arm is frozen at its
10k checkpoint (best available; default --timetrim-ref-step 10000).

train48 selection comes from the preroll fit48 manifest (same frozen split
record as the 10k round); dev12 is the fixed 12-episode list.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

DEV12_EPISODES = (9, 11, 17, 25, 26, 28, 30, 31, 43, 52, 55, 59)
PHASES = ("approach", "grasp", "lift_place", "release")
FAILURE_MARKERS = re.compile(r"\b(nan|inf)\b|out of memory|traceback", re.IGNORECASE)
ARM = 5


def departure(chunk: np.ndarray, baseline: np.ndarray, threshold: float) -> int | None:
    moved = np.max(np.abs(chunk[:, :ARM] - baseline[None, :ARM]), axis=1) >= threshold
    indices = np.flatnonzero(moved)
    return int(indices[0]) if len(indices) else None


def pair_metrics(pred_h0: np.ndarray, baseline: np.ndarray, desired: np.ndarray, threshold: float) -> dict:
    """h0 prediction vs desired target, both relative to baseline, in raw action units."""
    pred_delta = pred_h0[:ARM] - baseline[:ARM]
    desired_delta = desired[:ARM] - baseline[:ARM]
    desired_norm = float(np.linalg.norm(desired_delta))
    pred_norm = float(np.linalg.norm(pred_delta))
    denominator = desired_norm * pred_norm
    return {
        "departure_h": 0 if np.max(np.abs(pred_delta)) >= threshold else None,
        "pan_direction_ok": bool(np.sign(pred_delta[0]) == np.sign(desired_delta[0]) and np.sign(desired_delta[0]) != 0),
        "motion_cosine": float(np.dot(desired_delta, pred_delta) / denominator) if denominator else np.nan,
        "motion_norm_ratio": float(pred_norm / desired_norm) if desired_norm else np.nan,
    }


def aggregate(per_episode: dict) -> dict:
    cosines = [m["motion_cosine"] for m in per_episode.values() if not np.isnan(m["motion_cosine"])]
    pans = [m["pan_direction_ok"] for m in per_episode.values() if m["pan_direction_ok"] is not None]
    ratios = [m["motion_norm_ratio"] for m in per_episode.values() if not np.isnan(m["motion_norm_ratio"])]
    deps = [m["departure_h"] for m in per_episode.values() if m["departure_h"] is not None]
    return {
        "departure_h_median": float(np.median(deps)) if deps else None,
        "departure_count": len(deps),
        "cosine_positive_count": int(np.sum(np.asarray(cosines) > 0)) if cosines else 0,
        "cosine_positive_rate": float(np.mean(np.asarray(cosines) > 0)) if cosines else None,
        "cosine_mean": float(np.mean(cosines)) if cosines else None,
        "pan_direction_count": int(np.sum(pans)) if pans else 0,
        "pan_direction_rate": float(np.mean(pans)) if pans else None,
        "norm_ratio_median": float(np.median(ratios)) if ratios else None,
    }


def load_npz(directory: Path, episode: int) -> dict[str, np.ndarray]:
    with np.load(directory / f"episode_{episode:03d}_chunks.npz") as data:
        return {key: data[key] for key in data.files}


def reset_observation_metrics(npz_dir: Path, orig_actions: dict[int, np.ndarray],
                              boundaries: dict[int, dict], threshold: float) -> dict:
    per_episode = {}
    for ep in DEV12_EPISODES:
        data = load_npz(npz_dir, ep)
        pred_h0 = data["prediction_raw"][0, 0]
        truth_chunk = data["truth_raw"][0]
        start = boundaries[ep]["start"]
        desired = orig_actions[ep][start]
        if not np.allclose(truth_chunk[0], desired):
            raise ValueError(f"ep{ep}: eval truth chunk row 0 != original action at S")
        baseline = np.median(orig_actions[ep][0:15], axis=0)
        per_episode[str(ep)] = pair_metrics(pred_h0, baseline, desired, threshold)
    return {"metrics": aggregate(per_episode), "episodes": per_episode}


def train48_metrics(npz_dir: Path, orig_actions: dict[int, np.ndarray],
                    boundaries: dict[int, dict], episodes: list[int], threshold: float) -> dict:
    """Frame0 h0 metrics plus the per-joint h0 bias (same as analyze_formal3_h0_bias.py)."""
    per_episode = {}
    for ep in episodes:
        data = load_npz(npz_dir, ep)
        pred_h0 = data["prediction_raw"][0, 0]
        truth_chunk = data["truth_raw"][0]
        start = boundaries[ep]["start"]
        desired = orig_actions[ep][start]
        if not np.allclose(truth_chunk[0], desired):
            raise ValueError(f"ep{ep}: eval truth chunk row 0 != original action at S")
        baseline = np.median(orig_actions[ep][0:15], axis=0)
        metrics = pair_metrics(pred_h0, baseline, desired, threshold)
        metrics["bias_deg_arm"] = (pred_h0[:ARM] - truth_chunk[0, :ARM]).tolist()
        per_episode[str(ep)] = metrics
    agg = aggregate(per_episode)
    bias = np.asarray([m["bias_deg_arm"] for m in per_episode.values()])
    agg["bias_deg_abs_mean_arm"] = float(np.mean(np.abs(bias)))
    agg["bias_deg_abs_mean_per_joint"] = np.mean(np.abs(bias), axis=0).tolist()
    return {"metrics": agg, "episodes": per_episode}


def trim_start_metrics(npz_dir: Path, threshold: float) -> dict:
    per_episode = {}
    for ep in DEV12_EPISODES:
        data = load_npz(npz_dir, ep)
        pred_h0 = data["prediction_raw"][0, 0]
        truth_chunk = data["truth_raw"][0]
        baseline = np.median(truth_chunk[:15], axis=0)
        gt_h = departure(truth_chunk, baseline, threshold)
        if gt_h is None:
            per_episode[str(ep)] = {"departure_h": None, "pan_direction_ok": None,
                                    "motion_cosine": np.nan, "motion_norm_ratio": np.nan}
        else:
            per_episode[str(ep)] = pair_metrics(pred_h0, baseline, truth_chunk[gt_h], threshold)
    return {"metrics": aggregate(per_episode), "episodes": per_episode}


def mean_error(errors: list[np.ndarray]) -> dict:
    values = np.concatenate(errors, axis=0)
    joint = values.mean(axis=0)
    return {"arm": float(joint[:ARM].mean()), "gripper": float(joint[ARM]), "per_joint": joint.tolist()}


def interval_metrics(
    npz_dir: Path,
    phases_by_episode: dict[int, np.ndarray],
    data_lo: int | dict[int, int],
    phase_lo: int | dict[int, int] | None = None,
) -> dict:
    all_errors, phase_errors = [], {p: [] for p in PHASES}
    for ep in DEV12_EPISODES:
        data = load_npz(npz_dir, ep)
        pred = data["prediction_raw"][:, 0]
        truth = data["truth_raw"][:, 0]
        if pred.shape != truth.shape:
            raise ValueError(f"ep{ep}: prediction/truth length mismatch in {npz_dir}")
        episode_data_lo = data_lo[ep] if isinstance(data_lo, dict) else data_lo
        effective_phase_lo = data_lo if phase_lo is None else phase_lo
        episode_phase_lo = effective_phase_lo[ep] if isinstance(effective_phase_lo, dict) else effective_phase_lo
        errors = np.abs(pred[episode_data_lo:] - truth[episode_data_lo:])
        phases = phases_by_episode[ep][episode_phase_lo:episode_phase_lo + len(errors)]
        if len(errors) != len(phases):
            raise ValueError(f"ep{ep}: phase labels do not match interval length")
        all_errors.append(errors)
        for phase in PHASES:
            mask = phases == phase
            if mask.any():
                phase_errors[phase].append(errors[mask])
    return {"all": mean_error(all_errors),
            "phases": {phase: mean_error(values) for phase, values in phase_errors.items()}}


def load_original_actions(original_root: Path, episodes: list[int]) -> dict[int, np.ndarray]:
    table = pq.read_table(original_root / "data/chunk-000/file-000.parquet",
                          columns=["episode_index", "frame_index", "action"])
    actions = {}
    for ep in episodes:
        rows = table.filter(pc.equal(table["episode_index"], ep)).sort_by("frame_index")
        actions[ep] = np.stack(rows["action"].to_numpy())
    return actions


def load_phases(preroll_root: Path) -> dict[int, np.ndarray]:
    table = pq.read_table(preroll_root / "meta/reviewed_phase_labels.parquet")
    phases = {}
    for ep in DEV12_EPISODES:
        rows = table.filter(pc.equal(table["episode_index"], ep)).sort_by("frame_index")
        phases[ep] = np.asarray(rows["phase"].to_pylist(), dtype=object)
    return phases


def gate6(campaign: Path, source_root: Path, manifest_name: str, train_log_name: str,
          train48_root: Path) -> dict:
    manifest = json.loads((source_root / f"meta/{manifest_name}").read_text())
    checks = manifest["checks"]
    log = (campaign / train_log_name).read_text(errors="ignore")
    eval_ok = ((campaign / "eval_dev12/evaluation.done").is_file()
               and (campaign / "eval_cross_on_timetrim/evaluation.done").is_file()
               and (train48_root / "evaluation.done").is_file())
    exit_codes = []
    for family in ("eval_dev12", "eval_cross_on_timetrim"):
        exit_codes += [int(path.read_text().strip()) for path in (campaign / family / "logs").glob("*.exit")]
    exit_codes += [int(path.read_text().strip()) for path in (train48_root / "logs").glob("*.exit")]
    return {
        "manifest_checks": {key: bool(value) for key, value in checks.items()},
        "manifest_checks_pass": all(checks.values()),
        "train_done_marker": (campaign / "train.done").is_file(),
        "train_log_failure_markers": bool(FAILURE_MARKERS.search(log)),
        "eval_done_markers": eval_ok,
        "eval_exit_codes_all_zero": all(code == 0 for code in exit_codes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--timetrim-campaign", type=Path, required=True)
    parser.add_argument("--preroll-root", type=Path, required=True)
    parser.add_argument("--timetrim-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--train48-root", type=Path, default=None)
    parser.add_argument("--steps", default="20000,40000,60000,80000,100000",
                        help="comma-separated checkpoint steps of the candidate")
    parser.add_argument("--final-step", type=int, default=100000,
                        help="checkpoint the gate verdicts apply to")
    parser.add_argument("--timetrim-ref-step", type=int, default=10000,
                        help="frozen time-trim reference checkpoint")
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--train-log-name", default="train_DET_RESET_PREROLL_100K.log")
    parser.add_argument("--out", type=Path, default=None,
                        help="json output (default: campaign/summary.json); md goes next to it")
    args = parser.parse_args()

    steps = [int(x) for x in args.steps.split(",") if x.strip()]
    if args.final_step not in steps:
        raise ValueError(f"--final-step {args.final_step} not in --steps")
    train48_root = args.train48_root if args.train48_root is not None else args.campaign / "eval_train48"
    tt_ref_tag = f"{args.timetrim_ref_step:06d}"

    dev12_root = args.campaign / "eval_dev12"
    cross_root = args.campaign / "eval_cross_on_timetrim"
    tt_dev12_root = args.timetrim_campaign / "eval_dev12"
    if not (dev12_root / "evaluation.done").is_file() or not (cross_root / "evaluation.done").is_file():
        raise FileNotFoundError("Candidate evaluation families are not complete")
    if not (tt_dev12_root / "evaluation.done").is_file():
        raise FileNotFoundError("Time-trim dev12 evaluation is not available for reference")
    if not (train48_root / "evaluation.done").is_file():
        raise FileNotFoundError("Candidate train48 evaluation is not complete")

    boundaries = {row["episode_index"]: row
                  for row in json.loads((args.timetrim_root / "meta/time_trim_manifest.json").read_text())["boundaries"]}
    train_episodes = [int(x) for x in json.loads(
        (args.preroll_root / "meta/formal3_fit48_manifest.json").read_text())["train"]]
    orig_actions = load_original_actions(args.original_root, [*DEV12_EPISODES, *train_episodes])
    phases = load_phases(args.preroll_root)
    tt_summary = json.loads((args.timetrim_campaign / "summary.json").read_text())
    if str(args.timetrim_ref_step) not in tt_summary["checkpoints"]:
        raise KeyError(f"time-trim summary has no checkpoint {args.timetrim_ref_step}")
    tt_ref = tt_summary["checkpoints"][str(args.timetrim_ref_step)]["initial_same_original_observation"]["timetrim"]

    source_manifest = json.loads((args.preroll_root / "meta/reset_preroll_manifest.json").read_text())
    prefixes = {row["episode_index"]: row["prefix_frames"] for row in source_manifest["episodes"]}

    report = {"threshold_degrees": args.threshold,
              "dev12_episodes": list(DEV12_EPISODES),
              "train48_episodes": train_episodes,
              "timetrim_ref_step": args.timetrim_ref_step,
              "checkpoints": {}}
    for step in steps:
        tag = f"{step:06d}"
        dev12_dir = dev12_root / f"RESET_PREROLL_{tag}"
        cross_dir = cross_root / f"RESET_PREROLL_ON_TIMETRIM_{tag}"
        train48_dir = train48_root / f"RESET_PREROLL_TRAIN48_{tag}"
        tt_dir = tt_dev12_root / f"TIMETRIM_{tt_ref_tag}"
        reset_obs = reset_observation_metrics(dev12_dir, orig_actions, boundaries, args.threshold)
        train48 = train48_metrics(train48_dir, orig_actions, boundaries, train_episodes, args.threshold)
        trim_start = {
            "preroll": trim_start_metrics(cross_dir, args.threshold),
            "timetrim": trim_start_metrics(tt_dir, args.threshold),
        }
        task = {
            "preroll": interval_metrics(dev12_dir, phases, data_lo=prefixes),
            "timetrim": interval_metrics(tt_dir, phases, data_lo=0, phase_lo=prefixes),
        }
        task_ratio = {
            "arm": task["preroll"]["all"]["arm"] / task["timetrim"]["all"]["arm"],
            "gripper": task["preroll"]["all"]["gripper"] / task["timetrim"]["all"]["gripper"],
            "phases": {phase: task["preroll"]["phases"][phase]["arm"] / task["timetrim"]["phases"][phase]["arm"]
                       for phase in PHASES},
        }
        verdicts = {
            "gate1_train48_direction": train48["metrics"]["cosine_positive_count"] >= 33,
            "gate2_dev12_direction": reset_obs["metrics"]["cosine_positive_count"] >= 9,
            "gate3_dev12_ratio": 0.5 <= reset_obs["metrics"]["norm_ratio_median"] <= 2.0,
            "gate4_train48_bias": train48["metrics"]["bias_deg_abs_mean_arm"] <= 2.5,
            "gate5_task_mae_ratio_1_2": all(value <= 1.2 for value in
                                             [task_ratio["arm"], task_ratio["gripper"], *task_ratio["phases"].values()]),
        }
        report["checkpoints"][str(step)] = {
            "dev12_reset_observation": reset_obs,
            "train48_reset_observation": train48,
            "trim_start_observation": trim_start,
            "task_interval_mae": {"values": task, "preroll_over_timetrim_ratio": task_ratio},
            "time_trim_reference_reset_obs": {
                "cosine_positive_rate": tt_ref["h0_motion_cosine_positive_rate"],
                "pan_direction_rate": tt_ref["pan_direction_rate"],
                "norm_ratio_median": tt_ref["h0_motion_norm_ratio_median"],
            },
            "verdicts": verdicts,
        }

    integrity = gate6(args.campaign, args.preroll_root, "reset_preroll_manifest.json",
                      args.train_log_name, train48_root)
    final = report["checkpoints"][str(args.final_step)]["verdicts"]
    final["gate6_integrity"] = (integrity["manifest_checks_pass"] and integrity["train_done_marker"]
                                and not integrity["train_log_failure_markers"] and integrity["eval_done_markers"]
                                and integrity["eval_exit_codes_all_zero"])
    report["integrity"] = integrity
    report[f"final_verdicts_{args.final_step}"] = final
    report[f"overall_pass_{args.final_step}"] = all(final.values())

    out_json = args.out if args.out is not None else args.campaign / "summary.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [f"# Formal3 preroll 100k (B1) evaluation", "",
             "Reset observation = original frame 0; baseline = median of original actions 0..14; "
             f"desired = original action at trim start S; threshold {args.threshold} deg; "
             f"time-trim reference frozen at {args.timetrim_ref_step}.", "",
             "| step | dev12 cosine+ (gate2 >=9/12) | dev12 pan | dev12 ratio med (gate3 0.5-2) |",
             "|---:|---:|---:|---:|"]
    for step in steps:
        r = report["checkpoints"][str(step)]["dev12_reset_observation"]["metrics"]
        lines.append(f"| {step} | {r['cosine_positive_count']}/12 | {r['pan_direction_count']}/12 | {r['norm_ratio_median']:.3f} |")
    lines += ["", "| step | train48 cosine+ (gate1 >=33/48) | train48 pan | train48 ratio | train48 |bias| (gate4 <=2.5) |",
              "|---:|---:|---:|---:|---:|"]
    for step in steps:
        r = report["checkpoints"][str(step)]["train48_reset_observation"]["metrics"]
        lines.append(f"| {step} | {r['cosine_positive_count']}/48 | {r['pan_direction_count']}/48 | "
                     f"{r['norm_ratio_median']:.3f} | {r['bias_deg_abs_mean_arm']:.2f} |")
    lines += ["", f"| step | trim-start cosine+ preroll | trim-start cosine+ timetrim ({args.timetrim_ref_step}) |",
              "|---:|---:|---:|"]
    for step in steps:
        item = report["checkpoints"][str(step)]["trim_start_observation"]
        lines.append(f"| {step} | {item['preroll']['metrics']['cosine_positive_count']}/12 | "
                     f"{item['timetrim']['metrics']['cosine_positive_count']}/12 |")
    lines += ["", "| step | model | task arm MAE (aligned frames K+) | gripper | arm ratio vs timetrim (gate5 <=1.2) |",
              "|---:|---|---:|---:|---:|"]
    for step in steps:
        item = report["checkpoints"][str(step)]
        task = item["task_interval_mae"]
        ratio = task["preroll_over_timetrim_ratio"]["arm"]
        for model in ("preroll", "timetrim"):
            mae = task["values"][model]["all"]
            ratio_cell = f"{ratio:.3f}" if model == "preroll" else "-"
            lines.append(f"| {step} | {model} | {mae['arm']:.4f} | {mae['gripper']:.4f} | {ratio_cell} |")
    lines += ["", f"| gate | {args.final_step} verdict |", "|---|---|"]
    gate_names = {"gate1_train48_direction": "1: train48 h0 direction >= 33/48",
                  "gate2_dev12_direction": "2: dev12 h0 direction >= 9/12",
                  "gate3_dev12_ratio": "3: dev12 h0 norm ratio in [0.5, 2.0]",
                  "gate4_train48_bias": "4: train48 h0 |bias| <= 2.5 deg",
                  "gate5_task_mae_ratio_1_2": "5: task-interval MAE ratio <= 1.2",
                  "gate6_integrity": "6: data/training integrity"}
    for key, name in gate_names.items():
        lines.append(f"| {name} | {'PASS' if final[key] else 'FAIL'} |")
    lines.append(f"| overall | {'PASS' if report[f'overall_pass_{args.final_step}'] else 'FAIL'} |")
    out_json.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_json)


if __name__ == "__main__":
    main()
