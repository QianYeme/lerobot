#!/usr/bin/env python3
"""Read-only F0 analysis of legacy SO-101 gripper proprioception.

The script deliberately keeps human-reviewed boundaries and automatically
derived coarse phases separate. It never rewrites the source dataset or phase
annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PHASES = ("free_open", "closing_contact", "stable_grasp", "lift_place", "release")
SIGNALS = ("gripper_pos", "gripper_load", "gripper_current", "gripper_action")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_number(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def robust_summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "median": None, "iqr": None, "p10": None, "p90": None}
    q10, q25, q50, q75, q90 = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "n": int(len(values)),
        "median": float(q50),
        "iqr": float(q75 - q25),
        "p10": float(q10),
        "p90": float(q90),
    }


def effect_and_overlap(a: np.ndarray, b: np.ndarray) -> dict[str, float | None]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return {"cohens_d": None, "histogram_overlap": None}
    pooled = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    d = (b.mean() - a.mean()) / pooled if pooled > 0 else 0.0
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if hi == lo:
        overlap = 1.0
    else:
        bins = np.linspace(lo, hi, 41)
        ha, _ = np.histogram(a, bins=bins, density=True)
        hb, _ = np.histogram(b, bins=bins, density=True)
        overlap = float(np.minimum(ha, hb).sum() * (bins[1] - bins[0]))
    return {"cohens_d": float(d), "histogram_overlap": overlap}


def phase_ranges(boundaries: dict[str, int], length: int, free_window: int) -> dict[str, tuple[int, int]]:
    closing_start = int(boundaries["closing_start"])
    closing_end = int(boundaries["closing_end"])
    lift_start = int(boundaries["lift_start"])
    release_start = int(boundaries["release_start"])
    release_stop = min(length, int(boundaries["release_end"]) + 1)
    return {
        "free_open": (max(0, closing_start - free_window), closing_start),
        "closing_contact": (closing_start, closing_end),
        "stable_grasp": (closing_end, lift_start),
        "lift_place": (lift_start, release_start),
        "release": (release_start, release_stop),
    }


def window_features(values: np.ndarray, width: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {name: float("nan") for name in ("mean", "std", "slope", "delta", "min", "max")}
    tail = values[-min(width, len(values)) :]
    x = np.arange(len(tail), dtype=float)
    slope = float(np.polyfit(x, tail, 1)[0]) if len(tail) > 1 else 0.0
    return {
        "mean": float(tail.mean()),
        "std": float(tail.std()),
        "slope": slope,
        "delta": float(tail[-1] - tail[0]),
        "min": float(tail.min()),
        "max": float(tail.max()),
    }


def load_data(dataset: Path, annotations_path: Path) -> tuple[pd.DataFrame, dict, dict]:
    info_path = dataset / "meta/info.json"
    data_path = dataset / "data/chunk-000/file-000.parquet"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    table = pq.read_table(data_path, columns=["observation.state", "action", "timestamp", "frame_index", "episode_index", "index"])
    frame = table.to_pandas()
    state = np.stack(frame.pop("observation.state").to_numpy())
    action = np.stack(frame.pop("action").to_numpy())
    state_names = info["features"]["observation.state"]["names"]
    action_names = info["features"]["action"]["names"]
    required_state = ["gripper.pos", "gripper.load", "gripper.curr"]
    missing = [name for name in required_state if name not in state_names]
    if missing:
        raise ValueError(f"Required state signals missing: {missing}")
    if "gripper.pos" not in action_names:
        raise ValueError("gripper action is missing")
    frame["gripper_pos"] = state[:, state_names.index("gripper.pos")]
    frame["gripper_load"] = state[:, state_names.index("gripper.load")]
    frame["gripper_current"] = state[:, state_names.index("gripper.curr")]
    frame["gripper_action"] = action[:, action_names.index("gripper.pos")]
    for index, name in enumerate(state_names[:5]):
        frame[f"arm_{name}"] = state[:, index]
    return frame, info, annotations


def assign_phases(frame: pd.DataFrame, annotations: dict, free_window: int) -> tuple[pd.DataFrame, list[dict]]:
    chunks = []
    boundary_rows = []
    for episode_text, record in annotations["episodes"].items():
        episode = int(episode_text)
        ep = frame[frame.episode_index == episode].copy().sort_values("frame_index")
        if len(ep) != int(record["length"]):
            raise ValueError(f"Episode {episode}: data length {len(ep)} != annotation {record['length']}")
        ep["phase"] = "unresolved"
        ranges = phase_ranges(record["boundaries"], len(ep), free_window)
        local_frame = ep["frame_index"].to_numpy()
        for phase, (start, stop) in ranges.items():
            ep.loc[(local_frame >= start) & (local_frame < stop), "phase"] = phase
        chunks.append(ep)
        boundary_rows.append({"episode": episode, **record["boundaries"], "hold_frames": record["hold_frames"]})
    return pd.concat(chunks, ignore_index=True), boundary_rows


def analyse(frame: pd.DataFrame, window: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    selected = frame[frame.phase.isin(PHASES)].copy()
    phase_stats: dict[str, dict] = {}
    for phase in PHASES:
        phase_stats[phase] = {signal: robust_summary(selected.loc[selected.phase == phase, signal].to_numpy()) for signal in SIGNALS}

    episode_rows = []
    window_rows = []
    for (episode, phase), group in selected.groupby(["episode_index", "phase"], sort=True):
        row = {"episode": int(episode), "phase": phase, "frames": int(len(group))}
        for signal in SIGNALS:
            values = group[signal].to_numpy(dtype=float)
            row[f"{signal}_median"] = float(np.median(values))
            row[f"{signal}_range"] = float(values.max() - values.min())
            features = window_features(values, window)
            for feature, value in features.items():
                row[f"{signal}_window_{feature}"] = value
        episode_rows.append(row)
        window_rows.append(row.copy())
    episode_table = pd.DataFrame(episode_rows)
    window_table = pd.DataFrame(window_rows)

    comparisons = {}
    for target in ("closing_contact", "stable_grasp", "lift_place", "release"):
        comparisons[target] = {}
        for signal in ("gripper_pos", "gripper_load", "gripper_current"):
            free = selected.loc[selected.phase == "free_open", signal].to_numpy()
            other = selected.loc[selected.phase == target, signal].to_numpy()
            metric = effect_and_overlap(free, other)
            paired = episode_table.pivot(index="episode", columns="phase", values=f"{signal}_median")
            paired = paired.dropna(subset=["free_open", target])
            diff = paired[target] - paired["free_open"]
            metric["episode_sign_consistency"] = float(max((diff > 0).mean(), (diff < 0).mean())) if len(diff) else None
            metric["episode_median_delta"] = float(diff.median()) if len(diff) else None
            comparisons[target][signal] = metric

    temporal = {}
    for target in ("stable_grasp", "lift_place"):
        temporal[target] = {}
        for signal in ("gripper_pos", "gripper_load", "gripper_current"):
            single_col = f"{signal}_median"
            best = {"feature": "single_phase_median", **effect_and_overlap(
                episode_table.loc[episode_table.phase == "free_open", single_col].to_numpy(),
                episode_table.loc[episode_table.phase == target, single_col].to_numpy(),
            )}
            candidates = {}
            for feature in ("mean", "std", "slope", "delta", "min", "max"):
                col = f"{signal}_window_{feature}"
                value = effect_and_overlap(
                    window_table.loc[window_table.phase == "free_open", col].to_numpy(),
                    window_table.loc[window_table.phase == target, col].to_numpy(),
                )
                candidates[feature] = value
            ranked = [(name, value) for name, value in candidates.items() if value["cohens_d"] is not None]
            best_window_name, best_window = max(ranked, key=lambda item: abs(item[1]["cohens_d"]))
            temporal[target][signal] = {
                "single": best,
                "window_features": candidates,
                "best_window": {"feature": best_window_name, **best_window},
            }

    hold = selected[selected.phase == "lift_place"].copy()
    hold_rows = []
    for episode, group in hold.groupby("episode_index"):
        action = group.gripper_action.to_numpy(dtype=float)
        baseline = float(np.median(action[: min(window, len(action))]))
        # In this dataset the open pose has the larger gripper angle
        # (free/open ~= 41 deg; stable grasp ~= 21 deg). Reopening is therefore
        # an increase relative to the initially closed lift/place command.
        max_after = float(action.max())
        reopen_amount = max_after - baseline
        hold_rows.append({
            "episode": int(episode),
            "frames": int(len(action)),
            "action_range": float(action.max() - action.min()),
            "action_net_change": float(action[-1] - action[0]),
            "max_reopen_from_initial_closed": reopen_amount,
            "reopen_ge_5deg": bool(reopen_amount >= 5.0),
        })
    hold_table = pd.DataFrame(hold_rows)
    hold_summary = {
        "episodes": int(len(hold_table)),
        "action_range": robust_summary(hold_table.action_range.to_numpy()),
        "net_change": robust_summary(hold_table.action_net_change.to_numpy()),
        "max_reopen_from_initial_closed": robust_summary(hold_table.max_reopen_from_initial_closed.to_numpy()),
        "reopen_ge_5deg_episodes": int(hold_table.reopen_ge_5deg.sum()),
        "reopen_ge_5deg_rate": float(hold_table.reopen_ge_5deg.mean()),
    }

    quality = {}
    for signal in SIGNALS:
        values = frame[signal].to_numpy(dtype=float)
        finite = np.isfinite(values)
        unique, counts = np.unique(values[finite], return_counts=True)
        quality[signal] = {
            "frames": int(len(values)),
            "missing_or_nonfinite": int((~finite).sum()),
            "unique_values": int(len(unique)),
            "min": finite_number(values[finite].min()) if finite.any() else None,
            "max": finite_number(values[finite].max()) if finite.any() else None,
            "global_extreme_rate": float((counts[0] + counts[-1]) / finite.sum()) if len(counts) > 1 else 1.0,
        }

    metrics = {
        "phase_statistics": phase_stats,
        "free_vs_phase": comparisons,
        "temporal_separability": temporal,
        "hold_action_quality": hold_summary,
        "signal_quality": quality,
    }
    return metrics, episode_table, hold_table


def choose_representative(episode_table: pd.DataFrame) -> int:
    pivot = episode_table.pivot(index="episode", columns="phase", values="gripper_load_median")
    columns = [phase for phase in PHASES if phase in pivot]
    centered = pivot[columns] - pivot[columns].median()
    mad = centered.abs().median().replace(0, 1)
    normalized = centered / mad
    return int(normalized.abs().sum(axis=1).idxmin())


def plot_figures(frame: pd.DataFrame, episode_table: pd.DataFrame, hold_table: pd.DataFrame, figures: Path, representative: int) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    colors = {"free_open": "#4c78a8", "closing_contact": "#f58518", "stable_grasp": "#eeca3b", "lift_place": "#54a24b", "release": "#e45756"}

    ep = frame[frame.episode_index == representative].sort_values("frame_index")
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    for ax, signal in zip(axes, SIGNALS, strict=True):
        ax.plot(ep.frame_index, ep[signal], linewidth=1.2)
        ax.set_ylabel(signal.replace("gripper_", ""))
        ax.grid(alpha=0.2)
        for phase, group in ep[ep.phase.isin(PHASES)].groupby("phase"):
            ax.axvspan(group.frame_index.min(), group.frame_index.max(), color=colors[phase], alpha=0.10)
    axes[-1].set_xlabel("episode-local frame")
    fig.suptitle(f"Representative episode {representative}; coarse phases anchored to human-reviewed boundaries")
    fig.tight_layout()
    fig.savefig(figures / "representative_timeline.png", dpi=180)
    plt.close(fig)

    aligned = frame[frame.phase.isin(PHASES)].copy()
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for ax, signal in zip(axes, ("gripper_pos", "gripper_load", "gripper_current"), strict=True):
        for episode, group in aligned.groupby("episode_index"):
            anchor = int(group.loc[group.phase == "closing_contact", "frame_index"].min())
            x = group.frame_index.to_numpy() - anchor
            mask = (x >= -60) & (x <= 240)
            ax.plot(x[mask], group[signal].to_numpy()[mask], alpha=0.10, linewidth=0.8, color="#4c78a8")
        ax.axvline(0, color="black", linestyle="--", linewidth=1, label="closing start")
        ax.set_ylabel(signal.replace("gripper_", ""))
        ax.grid(alpha=0.2)
    axes[0].legend()
    axes[-1].set_xlabel("frames relative to human-reviewed closing start")
    fig.suptitle("All 60 episodes aligned at closing start")
    fig.tight_layout()
    fig.savefig(figures / "aligned_phase_timeline.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for ax, signal in zip(axes, ("gripper_load", "gripper_current"), strict=True):
        data = [frame.loc[frame.phase == phase, signal].to_numpy() for phase in PHASES]
        ax.boxplot(data, labels=PHASES, showfliers=False)
        ax.set_ylabel(signal.replace("gripper_", ""))
        ax.grid(axis="y", alpha=0.2)
    axes[-1].tick_params(axis="x", rotation=20)
    fig.suptitle("Force-related signal distributions by coarse phase")
    fig.tight_layout()
    fig.savefig(figures / "phase_distribution.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(hold_table.action_range, bins=15, color="#4c78a8", alpha=0.8)
    axes[0].set_xlabel("lift/place gripper action range (deg)")
    axes[0].set_ylabel("episodes")
    axes[1].hist(hold_table.max_reopen_from_initial_closed, bins=15, color="#e45756", alpha=0.8)
    axes[1].axvline(5.0, color="black", linestyle="--", label="audit threshold (not PASS gate)")
    axes[1].set_xlabel("maximum reopen from initial lift/place command (deg)")
    axes[1].legend()
    fig.suptitle("Expert gripper command stability during combined lift/place interval")
    fig.tight_layout()
    fig.savefig(figures / "hold_action_analysis.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--free-window", type=int, default=60)
    parser.add_argument("--window", type=int, default=15)
    args = parser.parse_args()
    if args.free_window <= 0 or args.window <= 1:
        raise ValueError("Window sizes must be positive and temporal window > 1")

    args.result.mkdir(parents=True, exist_ok=True)
    args.raw.mkdir(parents=True, exist_ok=True)
    frame, info, annotations = load_data(args.dataset, args.annotations)
    frame, boundary_rows = assign_phases(frame, annotations, args.free_window)
    metrics, episode_table, hold_table = analyse(frame, args.window)
    representative = choose_representative(episode_table)
    plot_figures(frame, episode_table, hold_table, args.result / "figures", representative)

    data_path = args.dataset / "data/chunk-000/file-000.parquet"
    provenance = {
        "dataset": str(args.dataset.resolve()),
        "episodes": int(info["total_episodes"]),
        "frames": int(info["total_frames"]),
        "fps": int(info["fps"]),
        "state_names": info["features"]["observation.state"]["names"],
        "action_names": info["features"]["action"]["names"],
        "dataset_info_sha256": sha256(args.dataset / "meta/info.json"),
        "dataset_data_sha256": sha256(data_path),
        "annotations": str(args.annotations.resolve()),
        "annotations_sha256": sha256(args.annotations),
        "phase_semantics": {
            "human_reviewed_boundaries": ["closing_start", "closing_end", "lift_start", "release_start", "release_end"],
            "automatically_derived": {
                "free_open": f"last {args.free_window} frames before closing_start",
                "closing_contact": "closing_start:closing_end; S1/S2 unresolved",
                "stable_grasp": "closing_end:lift_start",
                "lift_place": "lift_start:release_start; S4/S5 unresolved",
                "release": "release_start:release_end inclusive",
            },
        },
        "success_failure_metadata": "absent",
        "representative_episode": representative,
        "window_frames": args.window,
    }
    metrics = {"provenance": provenance, **metrics}
    (args.result / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.raw / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    episode_table.to_csv(args.raw / "episode_phase_features.csv", index=False)
    hold_table.to_csv(args.raw / "hold_action_per_episode.csv", index=False)
    pd.DataFrame(boundary_rows).to_csv(args.raw / "human_reviewed_boundaries.csv", index=False)
    frame[["episode_index", "frame_index", "timestamp", "phase", *SIGNALS]].to_parquet(args.raw / "phase_signal_rows.parquet", index=False)
    print(json.dumps({"status": "ok", "representative_episode": representative, "result": str(args.result), "raw": str(args.raw)}))


if __name__ == "__main__":
    main()
