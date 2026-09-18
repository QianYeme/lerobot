"""Scan temporal-ensemble coefficients on already-captured chunks.

LeRobot's ACTTemporalEnsembler weights w_i = exp(-coeff * i) with w_0 the OLDEST
action, so a positive coefficient favours old predictions and a negative one
favours new ones. This reuses the replay helpers of offline_eval_act_sequence.py
and reports tracking error (MAE) against smoothness (first/second differences).

Usage:
  python scripts/scan_c50_sequence_ensemble.py --npz-dir <capture dir> \
      [--coeffs 0.0,0.03,-0.01,-0.03,-0.1,-0.3] [--output <json>]
"""

import argparse
import glob
import json

import numpy as np

from lerobot.scripts.offline_eval_act_sequence import ensemble_replay, trajectory_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-dir", required=True)
    parser.add_argument("--coeffs", default="0.0,0.03,-0.01,-0.03,-0.1,-0.3")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    coefficients = [float(c) for c in args.coeffs.split(",")]

    episodes = sorted(glob.glob(args.npz_dir + "/episode_*_chunks.npz"))
    if not episodes:
        raise FileNotFoundError("No captured chunks under " + args.npz_dir)
    per_coefficient = {c: [] for c in coefficients}
    demo_step = []
    demo_step2 = []
    for path in episodes:
        data = np.load(path)
        chunks = data["prediction"]
        truth = data["truth"][:, 0]
        demo_step.append(float(np.abs(np.diff(truth, axis=0))[:, :-1].mean()))
        demo_step2.append(float(np.abs(np.diff(truth, n=2, axis=0))[:, :-1].mean()))
        for coefficient in coefficients:
            per_coefficient[coefficient].append(trajectory_metrics(ensemble_replay(chunks, coefficient), truth))

    report = {"npz_dir": args.npz_dir, "episodes": len(episodes), "coefficients": {}}
    report["demonstration_step_arm"] = float(np.mean(demo_step))
    report["demonstration_second_delta_arm"] = float(np.mean(demo_step2))
    print("episodes=%d  demonstration_step_arm=%.4f  demonstration_step2_arm=%.4f"
          % (len(episodes), report["demonstration_step_arm"], report["demonstration_second_delta_arm"]))
    header = "%-8s %8s %8s %8s %8s %8s" % ("coeff", "mae_arm", "mae_grip", "step_arm", "step2_arm", "step_grip")
    print(header)
    for coefficient in coefficients:
        rows = per_coefficient[coefficient]
        entry = {
            "mae_arm": float(np.mean([r["mae"]["arm"] for r in rows])),
            "mae_grip": float(np.mean([r["mae"]["gripper"] for r in rows])),
            "step_arm": float(np.mean([r["step_delta"]["arm"] for r in rows])),
            "step2_arm": float(np.mean([r["second_delta"]["arm"] for r in rows])),
            "step_grip": float(np.mean([r["step_delta"]["gripper"] for r in rows])),
        }
        report["coefficients"][str(coefficient)] = entry
        print("%-8s %8.4f %8.4f %8.4f %8.4f %8.4f"
              % (coefficient, entry["mae_arm"], entry["mae_grip"], entry["step_arm"], entry["step2_arm"], entry["step_grip"]))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print("written", args.output)


if __name__ == "__main__":
    main()
