"""Freeze a train40-internal split; do not modify data or start training."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from prepare_c50_nomaster import vector_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    manifest = json.loads((args.dataset_root / "meta/nomaster_manifest.json").read_text())
    training = sorted(manifest["train"])
    validation = sorted(manifest["validation"])
    reviewed = [0, 20, 49]
    if len(training) != 40 or len(validation) != 10 or set(training) & set(validation):
        raise ValueError("Expected disjoint train40 / val10")
    if not set(reviewed).issubset(training):
        raise ValueError("Reviewed label episodes must remain in training")
    development = sorted(random.Random(1000).sample([ep for ep in training if ep not in reviewed], 8))
    fit = [ep for ep in training if ep not in development]
    data = pa.concat_tables([pq.read_table(path) for path in sorted((args.dataset_root / "data").glob("*/*.parquet"))])
    data = data.filter(pa.compute.is_in(data["episode_index"], value_set=pa.array(fit)))
    state = np.asarray(data["observation.state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(data["action"].to_pylist(), dtype=np.float32)
    if state.shape != (32 * 718, 8) or actions.shape != (32 * 718, 6):
        raise ValueError("Expected complete eight-state / six-action fit32 frames")
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError("Nonfinite training data")
    report = {"seed": 1000, "fit32": fit, "development8": development,
              "fixed_validation10": validation, "reviewed_mask_episodes_kept_in_fit": reviewed,
              "fit32_frames": len(data),
              "fit32_state_action_statistics": {"observation.state": vector_stats(state), "action": vector_stats(actions)},
              "normalization_requirement": "Use these fit32 statistics in pilot processors; do not use the existing train40 statistics",
              "formal_training_ready": False,
              "warning": "Preparation artifact only; dataset and processors have not been changed. Existing val10 is not a blind test."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
