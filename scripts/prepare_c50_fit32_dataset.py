"""Prepare the fit32 dataset variant used by phase-C short training.

Creates 数据集/formal1_C50_nomaster_fit32 sharing data/videos/annotations and
most meta files with the nomaster dataset via symlinks, but with its own
meta/stats.json recomputed from fit32 only (per dev_split_with_stats.json) and
a fit32 manifest. Writes fit32_preparation_ready.json consumed as the gate by
train_c50_nomaster.sh short mode.

Usage: python scripts/prepare_c50_fit32_dataset.py [--source-root ...] [--dev-split ...] [--target ...]
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

SHARED_META = ("episodes", "info.json", "tasks.parquet")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("数据集/formal1_C50_nomaster"))
    parser.add_argument("--dev-split", type=Path,
                        default=Path("outputs/mask_inject_phase3_20260918/dev_split_with_stats.json"))
    parser.add_argument("--target", type=Path, default=Path("数据集/formal1_C50_nomaster_fit32"))
    args = parser.parse_args()

    if args.target.exists():
        raise FileExistsError(f"Refusing to overwrite existing target {args.target}")
    dev = json.loads(args.dev_split.read_text(encoding="utf-8"))
    fit32, dev8 = dev["fit32"], dev["development8"]
    if len(fit32) != 32 or len(dev8) != 8 or set(fit32) & set(dev8):
        raise ValueError("Unexpected fit32/development8 split in dev split file")
    fit_stats = dev["fit32_state_action_statistics"]
    if set(fit_stats) != {"observation.state", "action"}:
        raise ValueError("Dev split statistics must cover exactly observation.state and action")

    source_stats = json.loads((args.source_root / "meta/stats.json").read_text(encoding="utf-8"))
    new_stats = dict(source_stats)
    for feature in ("observation.state", "action"):
        if list(source_stats[feature]) != list(fit_stats[feature]):
            raise ValueError(f"Statistic keys of {feature} differ from source stats.json")
        new_stats[feature] = fit_stats[feature]

    target_meta = args.target / "meta"
    target_meta.mkdir(parents=True)
    for entry in SHARED_META:
        os.symlink((args.source_root / "meta" / entry).resolve(), target_meta / entry)
    for entry in ("data", "videos", "annotations"):
        os.symlink((args.source_root / entry).resolve(), args.target / entry)
    (target_meta / "stats.json").write_text(json.dumps(new_stats, indent=2), encoding="utf-8")
    manifest = {"train": fit32, "development": dev8,
                "validation": dev["fixed_validation10"], "source": str(args.dev_split)}
    (target_meta / "nomaster_manifest_fit32.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    reloaded = json.loads((target_meta / "stats.json").read_text(encoding="utf-8"))
    for feature in ("observation.state", "action"):
        for stat in fit_stats[feature]:
            assert reloaded[feature][stat] == fit_stats[feature][stat], f"stats mismatch {feature}.{stat}"
    assert (target_meta / "episodes").is_symlink() and (target_meta / "episodes").exists()
    assert (args.target / "data").is_symlink() and (args.target / "data").exists()

    ready_dir = Path("outputs/formal1_C50_nomaster_phase1/preparation")
    ready_dir.mkdir(parents=True, exist_ok=True)
    ready = {
        "status": "passed",
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "stats_sha256": hashlib.sha256((target_meta / "stats.json").read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((target_meta / "nomaster_manifest_fit32.json").read_bytes()).hexdigest(),
        "dev_split_sha256": hashlib.sha256(args.dev_split.read_bytes()).hexdigest(),
        "fit32": fit32, "development8": dev8,
    }
    ready_path = ready_dir / "fit32_preparation_ready.json"
    ready_path.write_text(json.dumps(ready, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"target": str(args.target), "ready": str(ready_path), **ready},
                     ensure_ascii=False, indent=2))
    print("dev-split directives:", json.dumps({"normalization_requirement": dev.get("normalization_requirement"),
                                               "formal_training_ready": dev.get("formal_training_ready"),
                                               "warning": dev.get("warning")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
