"""Summarize the preregistered Formal3 reset-pair 10k comparison against the time-trim arm.

Gate semantics (identical observations / identical baselines as the time-trim round):
- Gates 1-3 (reset observation = original frame 0): baseline = median of the original
  episode's first 15 actions; desired = original action at S (trim start); the reset-pair
  model's h0 prediction comes from its own dev12 eval at frame 0, whose observation is
  exactly original frame 0. Reference values are the time-trim cross-eval on the same
  observations.
- Gate 4 (time-trim start observation = original frame S): baseline = median of the
  trimmed truth chunk rows 0..14; desired = first departure within the chunk. The
  reset-pair model is cross-inferred on the time-trim dataset (max-frames 2); the
  time-trim model's reference comes from its own dev12 eval at frame 0.
- Gate 5 (common task interval): frames [15:length) of both datasets hold identical
  observations; arm/gripper MAE is compared there, plus per-phase MAE.
- Gate 6: build-time manifest checks, training-log failure markers, eval completion.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

EPISODES = (9, 11, 17, 25, 26, 28, 30, 31, 43, 52, 55, 59)
STEPS = (2000, 4000, 6000, 8000, 10000)
PHASES = ("approach", "grasp", "lift_place", "release")
FAILURE_MARKERS = re.compile(r"\b(nan|inf)\b|out of memory|traceback", re.IGNORECASE)


def departure(chunk: np.ndarray, baseline: np.ndarray, threshold: float) -> int | None:
    moved = np.max(np.abs(chunk[:, :5] - baseline[None, :5]), axis=1) >= threshold
    indices = np.flatnonzero(moved)
    return int(indices[0]) if len(indices) else None


def pair_metrics(pred_h0: np.ndarray, baseline: np.ndarray, desired: np.ndarray, threshold: float) -> dict:
    """h0 prediction vs desired target, both relative to baseline, in raw action units."""
    pred_delta = pred_h0[:5] - baseline[:5]
    desired_delta = desired[:5] - baseline[:5]
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
    for ep in EPISODES:
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


def trim_start_metrics(npz_dir: Path, threshold: float) -> dict:
    per_episode = {}
    for ep in EPISODES:
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
    return {"arm": float(joint[:5].mean()), "gripper": float(joint[5]), "per_joint": joint.tolist()}


def interval_metrics(
    npz_dir: Path,
    phases_by_episode: dict[int, np.ndarray],
    data_lo: int | dict[int, int],
    phase_lo: int | dict[int, int] | None = None,
) -> dict:
    all_errors, phase_errors = [], {p: [] for p in PHASES}
    for ep in EPISODES:
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


def load_original_actions(original_root: Path) -> dict[int, np.ndarray]:
    table = pq.read_table(original_root / "data/chunk-000/file-000.parquet",
                          columns=["episode_index", "frame_index", "action"])
    actions = {}
    for ep in EPISODES:
        rows = table.filter(pc.equal(table["episode_index"], ep)).sort_by("frame_index")
        actions[ep] = np.stack(rows["action"].to_numpy())
    return actions


def load_phases(resetpair_root: Path) -> dict[int, np.ndarray]:
    table = pq.read_table(resetpair_root / "meta/reviewed_phase_labels.parquet")
    phases = {}
    for ep in EPISODES:
        rows = table.filter(pc.equal(table["episode_index"], ep)).sort_by("frame_index")
        phases[ep] = np.asarray(rows["phase"].to_pylist(), dtype=object)
    return phases


def gate6(campaign: Path, source_root: Path, manifest_name: str, train_log_name: str) -> dict:
    manifest = json.loads((source_root / f"meta/{manifest_name}").read_text())
    checks = manifest["checks"]
    log = (campaign / train_log_name).read_text(errors="ignore")
    eval_ok = ((campaign / "eval_dev12/evaluation.done").is_file()
               and (campaign / "eval_cross_on_timetrim/evaluation.done").is_file())
    dev12_exits = list((campaign / "eval_dev12/logs").glob("*.exit"))
    cross_exits = list((campaign / "eval_cross_on_timetrim/logs").glob("*.exit"))
    exit_codes = [int(path.read_text().strip()) for path in dev12_exits + cross_exits]
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
    parser.add_argument("--resetpair-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--timetrim-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--variant", choices=("resetpair", "preroll"), default="resetpair")
    args = parser.parse_args()

    dev12_root = args.campaign / "eval_dev12"
    cross_root = args.campaign / "eval_cross_on_timetrim"
    tt_dev12_root = args.timetrim_campaign / "eval_dev12"
    if not (dev12_root / "evaluation.done").is_file() or not (cross_root / "evaluation.done").is_file():
        raise FileNotFoundError("Reset-pair evaluation families are not complete")
    if not (tt_dev12_root / "evaluation.done").is_file():
        raise FileNotFoundError("Time-trim dev12 evaluation is not available for reference")

    boundaries = {row["episode_index"]: row
                  for row in json.loads((args.timetrim_root / "meta/time_trim_manifest.json").read_text())["boundaries"]}
    orig_actions = load_original_actions(args.original_root)
    phases = load_phases(args.resetpair_root)
    tt_summary = json.loads((args.timetrim_campaign / "summary.json").read_text())

    if args.variant == "preroll":
        source_manifest_name = "reset_preroll_manifest.json"
        source_manifest = json.loads((args.source_root / f"meta/{source_manifest_name}").read_text())
        prefixes = {row["episode_index"]: row["prefix_frames"] for row in source_manifest["episodes"]}
        candidate_tag = "RESET_PREROLL"
        cross_tag = "RESET_PREROLL_ON_TIMETRIM"
        candidate_name = "preroll"
        train_log_name = "train_DET_RESET_PREROLL_10K.log"
    else:
        source_manifest_name = "reset_pair_manifest.json"
        prefixes = {ep: 15 for ep in EPISODES}
        candidate_tag = "RESETPAIR"
        cross_tag = "RESETPAIR_ON_TIMETRIM"
        candidate_name = "resetpair"
        train_log_name = "logs/train_DET_RESETPAIR_10K.log"

    report = {"threshold_degrees": args.threshold, "episodes": list(EPISODES), "checkpoints": {}}
    for step in STEPS:
        tag = f"{step:06d}"
        resetpair_dir = dev12_root / f"{candidate_tag}_{tag}"
        cross_dir = cross_root / f"{cross_tag}_{tag}"
        timetrim_dir = tt_dev12_root / f"TIMETRIM_{tag}"
        reset_obs = reset_observation_metrics(resetpair_dir, orig_actions, boundaries, args.threshold)
        trim_start = {
            candidate_name: trim_start_metrics(cross_dir, args.threshold),
            "timetrim": trim_start_metrics(timetrim_dir, args.threshold),
        }
        task = {
            candidate_name: interval_metrics(resetpair_dir, phases, data_lo=prefixes),
            "timetrim": interval_metrics(timetrim_dir, phases, data_lo=0, phase_lo=prefixes),
        }
        supplementary = {candidate_name: interval_metrics(resetpair_dir, phases, data_lo=0)}
        task_ratio = {
            "arm": task[candidate_name]["all"]["arm"] / task["timetrim"]["all"]["arm"],
            "gripper": task[candidate_name]["all"]["gripper"] / task["timetrim"]["all"]["gripper"],
            "phases": {phase: task[candidate_name]["phases"][phase]["arm"] / task["timetrim"]["phases"][phase]["arm"]
                       for phase in PHASES},
        }
        tt_ref = tt_summary["checkpoints"][str(step)]["initial_same_original_observation"]["timetrim"]
        verdicts = {
            "gate1_reset_direction": reset_obs["metrics"]["cosine_positive_count"] >= 11,
            "gate2_reset_pan": reset_obs["metrics"]["pan_direction_count"] >= 9,
            "gate3_reset_ratio": 0.5 <= reset_obs["metrics"]["norm_ratio_median"] <= 2.0,
            "gate4_trim_start_no_regression": (trim_start[candidate_name]["metrics"]["cosine_positive_count"]
                                               >= trim_start["timetrim"]["metrics"]["cosine_positive_count"]),
            "gate5_task_mae_ratio_1_2": all(value <= 1.2 for value in
                                             [task_ratio["arm"], task_ratio["gripper"], *task_ratio["phases"].values()]),
        }
        report["checkpoints"][str(step)] = {
            "reset_observation": reset_obs,
            "trim_start_observation": trim_start,
            "task_interval_mae_frames15plus": {"values": task, "resetpair_over_timetrim_ratio": task_ratio},
            "full_trajectory_mae_supplementary": supplementary,
            "time_trim_reference_reset_obs": {
                "cosine_positive_rate": tt_ref["h0_motion_cosine_positive_rate"],
                "pan_direction_rate": tt_ref["pan_direction_rate"],
                "norm_ratio_median": tt_ref["h0_motion_norm_ratio_median"],
            },
            "verdicts": verdicts,
        }

    integrity = gate6(args.campaign, args.source_root, source_manifest_name, train_log_name)
    final = report["checkpoints"]["10000"]["verdicts"]
    final["gate6_integrity"] = (integrity["manifest_checks_pass"] and integrity["train_done_marker"]
                                and not integrity["train_log_failure_markers"] and integrity["eval_done_markers"]
                                and integrity["eval_exit_codes_all_zero"])
    report["integrity"] = integrity
    report["final_verdicts_10000"] = final
    report["overall_pass_10000"] = all(final.values())

    out_json = args.campaign / "summary.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [f"# Formal3 {candidate_name} P3 evaluation", "",
             "Reset observation = original frame 0; baseline = median of original actions 0..14; "
             f"desired = original action at trim start S; threshold {args.threshold} deg.", "",
             "| step | model | cosine+ (gate1 >=11/12) | pan (gate2 >=9/12) | norm ratio med (gate3 0.5-2) |",
             "|---:|---|---:|---:|---:|"]
    for step in STEPS:
        item = report["checkpoints"][str(step)]
        r = item["reset_observation"]["metrics"]
        lines.append(f"| {step} | {candidate_name} | {r['cosine_positive_count']}/12 | {r['pan_direction_count']}/12 | {r['norm_ratio_median']:.3f} |")
        ref = item["time_trim_reference_reset_obs"]
        lines.append(f"| {step} | timetrim (ref) | {round(ref['cosine_positive_rate'] * 12)}/12 | "
                     f"{round(ref['pan_direction_rate'] * 12)}/12 | {ref['norm_ratio_median']:.3f} |")
    lines += ["", f"| step | trim-start cosine+ {candidate_name} | trim-start cosine+ timetrim (gate4 no regression) |",
              "|---:|---:|---:|"]
    for step in STEPS:
        item = report["checkpoints"][str(step)]["trim_start_observation"]
        lines.append(f"| {step} | {item[candidate_name]['metrics']['cosine_positive_count']}/12 | "
                     f"{item['timetrim']['metrics']['cosine_positive_count']}/12 |")
    interval_label = "aligned frames K+" if args.variant == "preroll" else "frames 15+"
    lines += ["", f"| step | model | task arm MAE ({interval_label}) | gripper | arm ratio vs timetrim (gate5 <=1.2) |",
              "|---:|---|---:|---:|---:|"]
    for step in STEPS:
        item = report["checkpoints"][str(step)]
        task = item["task_interval_mae_frames15plus"]
        ratio = task["resetpair_over_timetrim_ratio"]["arm"]
        for model in (candidate_name, "timetrim"):
            mae = task["values"][model]["all"]
            ratio_cell = f"{ratio:.3f}" if model == candidate_name else "-"
            lines.append(f"| {step} | {model} | {mae['arm']:.4f} | {mae['gripper']:.4f} | {ratio_cell} |")
    lines += ["", "| gate | 10k verdict |", "|---|---|"]
    gate_names = {"gate1_reset_direction": "1: reset-obs direction >= 11/12",
                  "gate2_reset_pan": "2: reset-obs pan >= 9/12",
                  "gate3_reset_ratio": "3: reset-obs norm ratio in [0.5, 2.0]",
                  "gate4_trim_start_no_regression": "4: trim-start direction no regression",
                  "gate5_task_mae_ratio_1_2": "5: task-interval MAE ratio <= 1.2",
                  "gate6_integrity": "6: data/training integrity"}
    for key, name in gate_names.items():
        lines.append(f"| {name} | {'PASS' if final[key] else 'FAIL'} |")
    lines.append(f"| overall | {'PASS' if report['overall_pass_10000'] else 'FAIL'} |")
    (args.campaign / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_json)


if __name__ == "__main__":
    main()
