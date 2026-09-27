"""Freeze the formal3 48/12 split and create a train-statistics dataset view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def vector_stats(values: np.ndarray) -> dict[str, list]:
    values = np.asarray(values, dtype=np.float64)
    result = {
        "min": values.min(0), "max": values.max(0), "mean": values.mean(0),
        "std": values.std(0), "count": np.array([len(values)]),
    }
    result.update({f"q{int(q * 100):02d}": np.quantile(values, q, axis=0)
                   for q in (0.01, 0.1, 0.5, 0.9, 0.99)})
    return {key: value.tolist() for key, value in result.items()}


def hardlink_tree(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if path.is_dir():
            (destination / relative).mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            (destination / relative).parent.mkdir(parents=True, exist_ok=True)
            os.link(path, destination / relative)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    if args.target.exists() or args.output.exists():
        raise FileExistsError("Refusing to overwrite target dataset view or split record")

    info = json.loads((args.source_root / "meta/info.json").read_text(encoding="utf-8"))
    if info["total_episodes"] != 60 or info["total_frames"] != 35917:
        raise ValueError("Unexpected formal3 dataset identity")
    state_names = info["features"]["observation.state"]["names"]
    if len(state_names) != 8 or "master_gripper.pos" in state_names:
        raise ValueError("Split source is not the frozen NOMASTER dataset")
    if not (args.source_root / "annotations/manifest.json").is_file():
        raise FileNotFoundError("Detection annotation Gate manifest is missing")

    rng = np.random.default_rng(args.seed)
    development = sorted(rng.choice(60, 12, replace=False).tolist())
    train = sorted(set(range(60)) - set(development))
    data = pq.read_table(args.source_root / "data/chunk-000/file-000.parquet")
    mask = np.isin(data["episode_index"].to_numpy(), train)
    stats = json.loads((args.source_root / "meta/stats.json").read_text(encoding="utf-8"))
    for feature in ("observation.state", "action"):
        stats[feature] = vector_stats(np.asarray(data[feature].to_pylist())[mask])

    target_meta = args.target / "meta"
    target_meta.mkdir(parents=True)
    hardlink_tree(args.source_root / "meta/episodes", target_meta / "episodes")
    for entry in ("info.json", "tasks.parquet"):
        os.link(args.source_root / "meta" / entry, target_meta / entry)
    for entry in ("data", "videos", "annotations"):
        hardlink_tree(args.source_root / entry, args.target / entry)
    (target_meta / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "train": train,
        "development": development,
        "train_frames": int(mask.sum()),
        "development_frames": int((~mask).sum()),
        "statistics_episodes": train,
        "normalization": "state/action recomputed from train only; images use ImageNet stats",
        "source_root": str(args.source_root.resolve()),
    }
    manifest_path = target_meta / "formal3_fit48_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "status": "passed",
        **manifest,
        "stats_sha256": hashlib.sha256((target_meta / "stats.json").read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
