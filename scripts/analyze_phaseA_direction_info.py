"""Phase A direction-information diagnosis (zero training cost).

Answers three questions with existing artifacts:
1. Does the diffusion 100k sampling distribution at reset frame 0 cover the
   per-episode desired first-motion direction? (multimodality hypothesis)
2. Does the deterministic DET 100k chunk's first significant motion point in
   the right direction? (direction info present in the representation?)
3. Does the preroll 10k h0 pan use the cup position? (information-flow audit:
   predicted/target pan vs top-camera box cx at frame 0)

Conventions (same as analyze_formal3_h0_bias.py): baseline = median of the
original episode's first 15 actions, desired = original action[S] where S is
the time-trim start; all values are natively degrees. The first significant
motion row r* is the first chunk row whose arm L-infinity deviation from the
baseline reaches 5 degrees.
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

ARM = 5
MOTION_DEG = 5.0


def load_npz(directory: Path, episode: int) -> dict[str, np.ndarray]:
    with np.load(directory / f"episode_{episode:03d}_chunks.npz") as data:
        return {key: data[key] for key in data.files}


def load_samples_npz(directory: Path, episode: int) -> dict[str, np.ndarray]:
    with np.load(directory / f"episode_{episode:03d}_samples.npz") as data:
        return {key: data[key] for key in data.files}


def load_original_actions(original_root: Path, episodes: list[int]) -> dict[int, np.ndarray]:
    table = pq.read_table(original_root / "data/chunk-000/file-000.parquet",
                          columns=["episode_index", "frame_index", "action"])
    actions = {}
    for ep in episodes:
        rows = table.filter(pc.equal(table["episode_index"], ep)).sort_by("frame_index")
        actions[ep] = np.stack(rows["action"].to_numpy())
    return actions


def frame0_box(ann_dir: Path, episode: int) -> tuple[float, float]:
    root = ET.parse(ann_dir / f"episode_{episode:03d}.xml").getroot()
    for box in root.iter("box"):
        if box.get("frame") == "0":
            cx = (float(box.get("xtl")) + float(box.get("xbr"))) / 2.0
            cy = (float(box.get("ytl")) + float(box.get("ybr"))) / 2.0
            return cx, cy
    raise ValueError(f"episode {episode}: no frame-0 box")


def first_motion(chunk: np.ndarray, baseline: np.ndarray) -> tuple[int, np.ndarray] | None:
    """First chunk row with arm L-inf deviation >= MOTION_DEG; None if absent."""
    for row in range(chunk.shape[0]):
        if np.max(np.abs(chunk[row, :ARM] - baseline[:ARM])) >= MOTION_DEG:
            return row, chunk[row, :ARM] - baseline[:ARM]
    return None


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else np.nan


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, default=None)
    parser.add_argument("--samples-name", default="diffusion_beforeS")
    parser.add_argument("--det-dir", type=Path, default=None)
    parser.add_argument("--preroll-dir", type=Path, default=None)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--timetrim-root", type=Path, required=True)
    parser.add_argument("--ann-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=str, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    episodes = [int(x) for x in args.episodes.split(",") if x.strip()]
    boundaries = {row["episode_index"]: row
                  for row in json.loads((args.timetrim_root / "meta/time_trim_manifest.json").read_text())["boundaries"]}
    orig_actions = load_original_actions(args.original_root, episodes)

    per_episode = {}
    for ep in episodes:
        start = boundaries[ep]["start"]
        baseline = np.median(orig_actions[ep][0:15], axis=0)
        desired_delta = orig_actions[ep][start][:ARM] - baseline[:ARM]
        desired_unit = desired_delta / (np.linalg.norm(desired_delta) or np.nan)
        cx, cy = frame0_box(args.ann_dir, ep)
        entry: dict = {"S": start, "desired_pan": float(desired_delta[0]),
                       "box_cx": cx, "box_cy": cy}

        if args.samples_dir is not None:
            samples = load_samples_npz(args.samples_dir, ep)["samples_raw"]
            directions, r_stars, pans = [], [], []
            for s in range(samples.shape[0]):
                found = first_motion(samples[s], baseline)
                if found is None:
                    continue
                r_star, delta = found
                directions.append(delta / np.linalg.norm(delta))
                r_stars.append(r_star)
                pans.append(delta[0])
            n = len(directions)
            entry["samples"] = {
                "n_samples": samples.shape[0], "n_with_motion": n,
                "distinct_pan_signs": len({1 if p > 0 else -1 if p < 0 else 0 for p in pans}),
                "pairwise_cosine_mean": float(np.mean(
                    [cosine(directions[i], directions[j])
                     for i in range(n) for j in range(i + 1, n)])) if n > 1 else np.nan,
                "best_cosine": max((cosine(d, desired_unit) for d in directions), default=np.nan),
                "mean_cosine": float(np.mean([cosine(d, desired_unit) for d in directions])) if n else np.nan,
                "coverage_05": bool(max((cosine(d, desired_unit) for d in directions), default=-1.0) >= 0.5),
                "coverage_07": bool(max((cosine(d, desired_unit) for d in directions), default=-1.0) >= 0.7),
                "r_star_median": float(np.median(r_stars)) if r_stars else np.nan,
                "pan_sign_agree_fraction": float(np.mean(
                    [1 if np.sign(p) == np.sign(desired_delta[0]) else 0 for p in pans])) if pans else np.nan,
            }

        if args.det_dir is not None:
            chunk = load_npz(args.det_dir, ep)["prediction_raw"][0]
            found = first_motion(chunk, baseline)
            if found is None:
                entry["det"] = {"r_star": None, "cosine": np.nan, "pan": np.nan}
            else:
                r_star, delta = found
                entry["det"] = {"r_star": r_star, "cosine": cosine(delta, desired_unit),
                                "pan": float(delta[0])}

        if args.preroll_dir is not None:
            pred_h0 = load_npz(args.preroll_dir, ep)["prediction_raw"][0, 0]
            entry["preroll_h0"] = {"cosine": cosine(pred_h0[:ARM] - baseline[:ARM], desired_unit),
                                   "pan": float(pred_h0[0] - baseline[0])}

        per_episode[str(ep)] = entry

    values = list(per_episode.values())
    target_pans = np.asarray([v["desired_pan"] for v in values])
    box_cxs = np.asarray([v["box_cx"] for v in values])

    report: dict = {"episodes": episodes, "sanity": {
        "corr_target_pan_vs_box_cx": pearson(target_pans, box_cxs),
        "box_cx_range": [float(np.min(box_cxs)), float(np.max(box_cxs))]}}
    if args.samples_dir is not None:
        s = [v["samples"] for v in values]
        report[f"samples_{args.samples_name}"] = {
            "coverage_05_count": int(np.sum([e["coverage_05"] for e in s])),
            "coverage_07_count": int(np.sum([e["coverage_07"] for e in s])),
            "best_cosine_mean": float(np.nanmean([e["best_cosine"] for e in s])),
            "mean_cosine_mean": float(np.nanmean([e["mean_cosine"] for e in s])),
            "pairwise_cosine_mean": float(np.nanmean([e["pairwise_cosine_mean"] for e in s])),
            "r_star_median": float(np.nanmedian([e["r_star_median"] for e in s])),
            "pan_sign_agree_fraction_mean": float(np.nanmean([e["pan_sign_agree_fraction"] for e in s])),
        }
    if args.det_dir is not None:
        d = [v["det"] for v in values]
        pans = np.asarray([e["pan"] for e in d if not np.isnan(e["pan"])])
        report["det_chunk"] = {
            "cosine_positive_count": int(np.sum([e["cosine"] > 0 for e in d])),
            "cosine_mean": float(np.nanmean([e["cosine"] for e in d])),
            "r_star_median": float(np.nanmedian([e["r_star"] for e in d])),
            "corr_pan_vs_box_cx": pearson(pans, box_cxs[:len(pans)]),
            "corr_pan_vs_target_pan": pearson(pans, target_pans[:len(pans)]),
        }
    if args.preroll_dir is not None:
        p = [v["preroll_h0"] for v in values]
        pans = np.asarray([e["pan"] for e in p])
        report["preroll_h0"] = {
            "cosine_positive_count": int(np.sum([e["cosine"] > 0 for e in p])),
            "cosine_mean": float(np.nanmean([e["cosine"] for e in p])),
            "corr_pan_vs_box_cx": pearson(pans, box_cxs),
            "corr_pan_vs_target_pan": pearson(pans, target_pans),
        }

    report["per_episode"] = per_episode
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [f"# Phase A direction-info diagnosis: {args.out.parent.name}/{args.out.name}", "",
             f"- episodes: {len(episodes)}",
             f"- sanity corr(target pan, box cx) = {report['sanity']['corr_target_pan_vs_box_cx']:.3f} "
             f"(box cx range {report['sanity']['box_cx_range'][0]:.0f}–{report['sanity']['box_cx_range'][1]:.0f})"]
    if f"samples_{args.samples_name}" in report:
        m = report[f"samples_{args.samples_name}"]
        lines += ["", f"## samples probe: {args.samples_name}",
                  f"- coverage>=0.5: {m['coverage_05_count']}/{len(episodes)} | coverage>=0.7: {m['coverage_07_count']}/{len(episodes)}",
                  f"- best cosine mean {m['best_cosine_mean']:.3f} | mean cosine {m['mean_cosine_mean']:.3f}",
                  f"- pairwise cosine mean (diversity) {m['pairwise_cosine_mean']:.3f}",
                  f"- first-motion row median {m['r_star_median']:.1f} | pan sign agree {m['pan_sign_agree_fraction_mean']:.2f}"]
    if "det_chunk" in report:
        m = report["det_chunk"]
        lines += ["", "## DET 100k chunk first-motion direction",
                  f"- cosine+ {m['cosine_positive_count']}/{len(episodes)} | mean {m['cosine_mean']:.3f} | r* median {m['r_star_median']:.1f}",
                  f"- corr(pan, box cx) {m['corr_pan_vs_box_cx']:.3f} | corr(pan, target pan) {m['corr_pan_vs_target_pan']:.3f}"]
    if "preroll_h0" in report:
        m = report["preroll_h0"]
        lines += ["", "## preroll 10k h0 (existing eval)",
                  f"- cosine+ {m['cosine_positive_count']}/{len(episodes)} | mean {m['cosine_mean']:.3f}",
                  f"- corr(pan, box cx) {m['corr_pan_vs_box_cx']:.3f} | corr(pan, target pan) {m['corr_pan_vs_target_pan']:.3f}"]
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
