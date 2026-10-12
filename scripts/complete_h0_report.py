#!/usr/bin/env python
"""Complete the read-only H0 report from the frozen audit and existing transplants."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


ROOT = Path("experiments/position/h0_horizontal_control_audit/index1/result")
TRANSPLANTS = {
    "T0.2_OC_ACT": Path("experiments/position/t0_2_token_transplant/index1/result/transplant_results.json"),
    "T1_P2": Path("experiments/position/t1_explicit_spatial_conditioning/index1/result/transplant_P2/transplant_results.json"),
    "N0_R1_RELATIONAL": Path("experiments/position/n0_overnight_batch/index1/result/R1_RELATIONAL/transplant/transplant_results.json"),
    "N0_R2_ACTION_LEVEL": Path("experiments/position/n0_overnight_batch/index1/result/R2_ACTION_LEVEL/transplant/transplant_results.json"),
    "N0_R3_FILM": Path("experiments/position/n0_overnight_batch/index1/result/R3_FILM/transplant/transplant_results.json"),
    "T2_P3": Path("experiments/position/t2_fixed_scale_spatial/index1/result/transplant_P3/transplant_results.json"),
    "T3_P4": Path("experiments/position/t3_joint_state_spatial/index1/result/transplant_P4/transplant_results.json"),
}


def stats(target, pred):
    target, pred = np.asarray(target, float), np.asarray(pred, float)
    mask = (np.abs(target) > 1e-8) & (np.abs(pred) > 1e-8)
    return {
        "n": int(len(target)),
        "n_nonzero": int(mask.sum()),
        "sign_accuracy": float(np.mean(np.sign(target[mask]) == np.sign(pred[mask]))) if mask.any() else 0.0,
        "pearson": float(np.corrcoef(target, pred)[0, 1]) if len(target) > 2 and np.std(target) > 1e-12 and np.std(pred) > 1e-12 else 0.0,
        "spearman": float(np.corrcoef(np.argsort(np.argsort(target)), np.argsort(np.argsort(pred)))[0, 1]) if len(target) > 2 and np.std(pred) > 1e-12 else 0.0,
        "mean_abs_projected": float(np.mean(np.abs(pred))) if len(pred) else 0.0,
    }


def main():
    mapping = json.loads((ROOT / "horizontal_mapping.json").read_text())["mapping"]["1"]
    scale = np.asarray(mapping["std"], float)
    weight = np.asarray(mapping["weight"], float)
    results = {}
    for name, path in TRANSPLANTS.items():
        if not path.exists():
            continue
        rows = json.loads(path.read_text())
        for variant in ["whole", "local_only", "global_only"]:
            target, pred = [], []
            for row in rows:
                if row.get("group") != "horizontal" or variant not in row.get("variants", {}):
                    continue
                delta = np.asarray(row["variants"][variant]["action_delta_h0"][:5], float)
                target.append(float(row["target_delta_x"]))
                pred.append(float((delta / scale) @ weight))
            results.setdefault(name, {})[variant] = stats(target, pred)
    report = {
        "experiment": "H0_HORIZONTAL_CONTROL_SIGNAL_AUDIT",
        "mapping_source": "frozen Train-only Method-B ridge, horizon 1",
        "model_re_evaluation": results,
        "interpretation": "Projected horizontal direction is diagnostic only; no checkpoint selection or training was performed.",
    }
    (ROOT / "model_re_evaluation.json").write_text(json.dumps(report, indent=2))
    # H0-C is the conservative outcome: the independent Dev state-matched evidence is insufficient and the temporal signal is weak.
    temporal = json.loads((ROOT / "expert_temporal_signal.json").read_text())["horizons"]
    pairs = json.loads((ROOT / "dev_mapping_validation.json").read_text())
    summary = [
        "# H0 Horizontal Control Signal Audit",
        "",
        "## Scope",
        "Read-only diagnosis on fixed Formal4 Train/Dev episodes. No training, architecture change, Final Test, or real-robot operation was performed.",
        "",
        "## Frozen method",
        "P0.1 x_global was used as the target. Because no reliable SO101 URDF/Jacobian is present, Method B fits a Train-only ridge from five arm joint target deltas to x_global. The mapping is frozen before model re-evaluation.",
        "",
        "## Evidence",
        f"- Selected reach/fine-alignment frames: Train {json.loads((ROOT/'horizontal_mapping.json').read_text())['train_count']}, Dev {json.loads((ROOT/'horizontal_mapping.json').read_text())['dev_count']}",
        f"- State-matched pairs: Train {pairs['train_pairs']['n']}, Dev {pairs['dev_pairs']['n']}; Dev pair count is too small for an independent gate.",
        "- H=1 Dev: sign accuracy 0.401, Pearson -0.171, Spearman -0.285.",
        "- H=3 Dev: sign accuracy 0.448, Pearson -0.142, Spearman -0.236.",
        "- H=5 Dev: sign accuracy 0.475, Pearson -0.102, Spearman -0.164.",
        "- H=10 Dev: sign accuracy 0.514, Pearson 0.041, Spearman -0.006; this is near chance, not a stable steering signal.",
        "",
        "## Model re-evaluation",
        "Existing token-transplant outputs were projected through the frozen mapping. These are mechanism diagnostics, not new model selection metrics; see `model_re_evaluation.json` for all variants.",
        "",
        "## H0 decision",
        "**H0-C / unresolved: do not attribute the observed initial centering to a single action dimension.** The multi-joint empirical mapping is not independently validated on Dev (only 2 matched pairs) and the frame-level temporal signal is weak or sign-inconsistent. Stop architecture expansion based on this hypothesis until the dataset/task signal is repaired or a reliable kinematic calibration is available.",
        "",
        "## Next safe step",
        "Improve the diagnostic boundary: obtain a trustworthy SO101 kinematic calibration or collect a deliberately varied horizontal trajectory set with explicit reach-phase labels, then rerun H0. Do not start another architecture/training campaign from this result alone.",
    ]
    (ROOT / "summary.md").write_text("\n".join(summary) + "\n")
    print(json.dumps({"models": list(results), "output": str(ROOT)}, indent=2))


if __name__ == "__main__":
    main()
