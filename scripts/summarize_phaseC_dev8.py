"""Phase-C screening summary over dev8 chunk captures (C50_MASK_INJECT_优化方案 §8).

For every C50_{MODEL}_dev8_s{STEP} run under the given root, aggregate raw-unit
executed-trajectory statistics (queue_1 = chunk head) and the normalized chunk
statistics into the inputs of the engineering screening line: h0 MAE (arm/gripper),
adjacent chunk conflict, gripper-event delay, stage-wise error and motion ratio.
Prints comparison tables and writes summary.json next to the run directories.

Gripper events: the C50 command shape is rest(0) -> wide open -> grip level ->
release open -> rest, so the grasp/release frames are level crossings with the
threshold placed between the open plateau and the mid-episode grip level (median
of the middle fifth), not between the episode min and max.

Usage: summarize_phaseC_dev8.py --root outputs/act_sequence_phaseC_2026-09-18
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

PLATEAU = 10  # consecutive frames required to accept a level crossing
OPEN_FRACTION = 0.4  # threshold sits this far from the grip plateau towards the episode peak
RUN_PATTERN = re.compile(r"C50_(DET|I0|I1)_dev8_s(\d+)$")


def first_run(mask, start=0):
    run = 0
    for i in range(start, len(mask)):
        run = run + 1 if mask[i] else 0
        if run >= PLATEAU:
            return i - PLATEAU + 1
    return None


def gripper_events(signal, threshold=None):
    """Opening / grasp / release frame indices of a gripper command series."""
    if threshold is None:
        grip_level = float(np.median(signal[len(signal) * 2 // 5:len(signal) * 3 // 5]))
        threshold = grip_level + OPEN_FRACTION * (float(signal.max()) - grip_level)
    above = signal > threshold
    opening = first_run(above)
    grasp = first_run(~above, opening + PLATEAU) if opening is not None else None
    release = first_run(above, grasp + PLATEAU) if grasp is not None else None
    return {"threshold": threshold, "opening": opening, "grasp": grasp, "release": release}


def mean_or_none(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def summarize_run(run_dir):
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    per_episode = {}
    for ep, rep in sorted(results["episodes"].items(), key=lambda kv: int(kv[0])):
        data = np.load(run_dir / f"episode_{int(ep):03d}_chunks.npz")
        truth, pred = data["truth_raw"][:, 0], data["prediction_raw"][:, 0]
        row = {
            "h0_arm_raw": float(np.abs(pred[:, :-1] - truth[:, :-1]).mean()),
            "h0_gripper_raw": float(np.abs(pred[:, -1] - truth[:, -1]).mean()),
            "h0_arm_normalized": rep["normalized"]["h0_mae"]["arm"],
            "conflict_h1_5_arm": rep["normalized"]["chunk_conflict"]["h1_5"]["mean"]["arm"],
            "conflict_h1_5_gripper": rep["normalized"]["chunk_conflict"]["h1_5"]["mean"]["gripper"],
            "queue1_step2_arm": rep["normalized"]["execution_replay"]["queue_1"]["second_delta"]["arm"],
        }
        demo_step = np.abs(np.diff(truth, axis=0)).mean(axis=0)
        pred_step = np.abs(np.diff(pred, axis=0)).mean(axis=0)
        row["step_ratio_arm"] = float(pred_step[:-1].mean() / demo_step[:-1].mean())
        row["step_ratio_gripper"] = float(pred_step[-1] / demo_step[-1])
        demo = gripper_events(truth[:, -1])
        if demo["grasp"] is not None:
            predicted = gripper_events(pred[:, -1], threshold=demo["threshold"])
            row["delay_grasp_frames"] = None if predicted["grasp"] is None else int(predicted["grasp"] - demo["grasp"])
            row["delay_release_frames"] = None if predicted["release"] is None or demo["release"] is None \
                else int(predicted["release"] - demo["release"])
            release = demo["release"] if demo["release"] is not None else len(truth)
            stages = {"pre": slice(0, demo["grasp"]), "grasp": slice(demo["grasp"], release),
                      "post": slice(release, len(truth))}
            for name, sl in stages.items():
                if sl.stop - sl.start > 5:
                    row[f"stage_{name}_arm"] = float(np.abs(pred[sl, :-1] - truth[sl, :-1]).mean())
        per_episode[ep] = row
    keys = ["h0_arm_raw", "h0_gripper_raw", "h0_arm_normalized", "conflict_h1_5_arm", "conflict_h1_5_gripper",
            "queue1_step2_arm", "step_ratio_arm", "step_ratio_gripper",
            "stage_pre_arm", "stage_grasp_arm", "stage_post_arm", "delay_grasp_frames", "delay_release_frames"]
    aggregate = {key: mean_or_none([row.get(key) for row in per_episode.values()]) for key in keys}
    aggregate["episodes"] = len(per_episode)
    aggregate["delay_grasp_detected"] = sum(1 for row in per_episode.values() if row.get("delay_grasp_frames") is not None)
    aggregate["delay_release_detected"] = sum(1 for row in per_episode.values() if row.get("delay_release_frames") is not None)
    return {"run": run_dir.name, "episodes": per_episode, "aggregate": aggregate}


def relative_to_reference(row, base):
    def ratio(key):
        value, reference = row["aggregate"].get(key), base["aggregate"].get(key)
        return None if not value or not reference else value / reference
    return {"h0_arm": ratio("h0_arm_raw"), "h0_gripper": ratio("h0_gripper_raw"),
            "conflict_h1_5_arm": ratio("conflict_h1_5_arm"), "conflict_h1_5_gripper": ratio("conflict_h1_5_gripper"),
            "stage_grasp_arm": ratio("stage_grasp_arm")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for run_dir in sorted(args.root.iterdir()):
        match = RUN_PATTERN.fullmatch(run_dir.name)
        if match and (run_dir / "results.json").exists() and (run_dir / "evaluation.done").exists():
            runs[(match.group(1), match.group(2))] = summarize_run(run_dir)
    columns = ["h0_arm_raw", "h0_gripper_raw", "conflict_h1_5_arm", "conflict_h1_5_gripper",
               "stage_pre_arm", "stage_grasp_arm", "stage_post_arm", "delay_grasp_frames", "delay_release_frames",
               "step_ratio_arm", "step_ratio_gripper", "queue1_step2_arm"]
    print("| run | " + " | ".join(columns) + " |")
    print("|" + "---|" * (len(columns) + 1))
    for (model, step), run in sorted(runs.items()):
        cells = [(f"{run['aggregate'][key]:.4f}" if run["aggregate"].get(key) is not None else "-")
                 for key in columns]
        print(f"| {model} s{step} | " + " | ".join(cells) + " |")
    print("\n### 相对同预算参照的比值（<1 表示优于参照；验收线：h0 ≤1.05，冲突或关键阶段 ≤0.90）")
    for (model, step), run in sorted(runs.items()):
        if model == "I1" and ("I0", step) in runs:
            base, label = runs[("I0", step)], "I1/I0"
        elif model != "DET" and ("DET", step) in runs:
            base, label = runs[("DET", step)], f"{model}/DET"
        else:
            continue
        ratios = relative_to_reference(run, base)
        text = "  ".join(f"{key}={value:.3f}" for key, value in ratios.items() if value is not None)
        print(f"- {label} s{step}: {text}")
    (args.root / "screen_summary.json").write_text(
        json.dumps({f"{model}_s{step}": run for (model, step), run in runs.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
