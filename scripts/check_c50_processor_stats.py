"""Verify a checkpoint's saved normalization stats equal the fit32 statistics.

The preprocessor normalizer must embed the fit32 observation.state mean/std and
the postprocessor unnormalizer the fit32 action mean/std; anything else means
the short run did not use the fit32 normalization as required by the plan.

Usage: python scripts/check_c50_processor_stats.py <pretrained_model_dir> <dev_split_json>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open


def load_tensors(path, filename):
    target = Path(path) / filename
    if not target.exists():
        return None
    tensors = {}
    with safe_open(str(target), framework="np") as handle:
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    return tensors


def match(feature, stat, reference, tensors, tolerance=1e-4):
    if tensors is None:
        return {"key": None, "max_abs_diff": None}
    for key, value in tensors.items():
        if value.shape == reference.shape and np.allclose(value, reference, atol=tolerance):
            return {"key": key, "max_abs_diff": float(np.abs(value - reference).max())}
    return {"key": None, "max_abs_diff": None}


def compare(checkpoint_dir, dev_split_json):
    dev = json.loads(Path(dev_split_json).read_text(encoding="utf-8"))
    fit = dev["fit32_state_action_statistics"]
    normalizer = load_tensors(checkpoint_dir, "policy_preprocessor_step_3_normalizer_processor.safetensors")
    unnormalizer = load_tensors(checkpoint_dir, "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    report = {"status": "passed", "checkpoint_dir": str(checkpoint_dir), "dev_split": str(dev_split_json)}
    expected = {}
    for feature, stats in fit.items():
        for stat in ("mean", "std"):
            expected[f"{feature}.{stat}"] = np.asarray(stats[stat], dtype=np.float64)
    report["normalizer"] = {name: match("observation.state", stat, expected[f"observation.state.{stat}"], normalizer)
                            for name, stat in (("mean", "mean"), ("std", "std"))}
    report["unnormalizer"] = {name: match("action", stat, expected[f"action.{stat}"], unnormalizer)
                              for name, stat in (("mean", "mean"), ("std", "std"))}
    for section in (report["normalizer"], report["unnormalizer"]):
        if any(entry["key"] is None for entry in section.values()):
            report["status"] = "failed"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("dev_split_json", type=Path)
    args = parser.parse_args()
    report = compare(args.checkpoint_dir, args.dev_split_json)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    sys.exit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
