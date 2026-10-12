#!/usr/bin/env python
"""Train-only P0.1 temperature sweep using the frozen P0 bank and frames."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from calibrate_p0_cup_prototypes import dino_patches, load_dino, patch_geometry


TAUS = (1.00, 0.20, 0.10, 0.05, 0.02)


def rank(values):
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def corr(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def measure(patches, bank, box, tau):
    sim = (F.normalize(patches, dim=-1) @ F.normalize(bank, dim=-1).T).max(dim=-1).values
    selected, inside, _, target_224 = patch_geometry(box)
    weights = torch.softmax(sim / tau, dim=0)
    coords = torch.linspace(-1, 1, 16, device=sim.device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    pred_x = float((weights * xx.flatten()).sum())
    pred_y = float((weights * yy.flatten()).sum())
    pred = ((pred_x + 1) * 320, (pred_y + 1) * 240)
    target = (target_224[0] * 640 / 224, target_224[1] * 480 / 224)
    error = float(np.hypot(pred[0] - target[0], pred[1] - target[1]))
    peak = int(sim.argmax())
    inside2 = inside.reshape(16, 16)
    neighbor = F.max_pool2d(inside2.float()[None, None], 3, 1, 1)[0, 0].bool().flatten()
    corner = torch.zeros(16, 16, dtype=torch.bool, device=sim.device)
    corner[:2, :2] = corner[:2, -2:] = True
    corner[-2:, :2] = corner[-2:, -2:] = True
    entropy = float(-(weights * (weights.clamp_min(1e-12).log())).sum())
    return {
        "soft_center_xy_pixels": list(pred),
        "target_center_xy_pixels": list(target),
        "error": error,
        "peak_in_box": bool(inside[peak]),
        "peak_in_box_or_neighbor": bool(neighbor[peak]),
        "inside_mass": float(weights[inside].sum()),
        "outside_mass": float(weights[~inside].sum()),
        "entropy": entropy,
        "effective_patch_count": float(np.exp(entropy)),
        "top1_weight": float(weights.max()),
        "top3_mass": float(torch.topk(weights, 3).values.sum()),
        "cup_max_similarity": float(sim[inside].max()),
        "corner_max_similarity": float(sim[corner.flatten()].max()),
        "similarity": sim.reshape(16, 16).detach().cpu(),
    }


def overlay(image, row, box, output):
    heat = row["similarity"].numpy()
    heat = ((heat - heat.min()) / max(float(heat.max() - heat.min()), 1e-8) * 255).astype(np.uint8)
    visual = cv2.addWeighted(image, 0.58, cv2.applyColorMap(cv2.resize(heat, (640, 480), interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_VIRIDIS), 0.42, 0)
    x1, y1, x2, y2 = box
    cv2.rectangle(visual, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.drawMarker(visual, tuple(round(x) for x in row["target_center_xy_pixels"]), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
    cv2.drawMarker(visual, tuple(round(x) for x in row["soft_center_xy_pixels"]), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
    cv2.imwrite(str(output), visual)


def summary(rows):
    errors = np.array([r["error"] for r in rows])
    return {
        "count": len(rows),
        "median_error_px": float(np.median(errors)),
        "p90_error_px": float(np.quantile(errors, 0.9)),
        "mean_error_px": float(errors.mean()),
        "peak_in_box_rate": float(np.mean([r["peak_in_box"] for r in rows])),
        "peak_in_box_or_neighbor_rate": float(np.mean([r["peak_in_box_or_neighbor"] for r in rows])),
        "inside_mass_mean": float(np.mean([r["inside_mass"] for r in rows])),
        "outside_mass_mean": float(np.mean([r["outside_mass"] for r in rows])),
        "entropy_mean": float(np.mean([r["entropy"] for r in rows])),
        "effective_patch_count_mean": float(np.mean([r["effective_patch_count"] for r in rows])),
        "top1_weight_mean": float(np.mean([r["top1_weight"] for r in rows])),
        "top3_mass_mean": float(np.mean([r["top3_mass"] for r in rows])),
        "cup_beats_corner_rate": float(np.mean([r["cup_max_similarity"] > r["corner_max_similarity"] for r in rows])),
    }


def main(args):
    result = Path(args.result_dir)
    manifest = json.loads((result / "reference_manifest.json").read_text(encoding="utf-8"))
    annotations = json.loads((result / "coarse_box_annotations.json").read_text(encoding="utf-8"))
    boxes = {(r["episode_index"], r["frame_index"]): r["coarse_box_xyxy"] for r in annotations["annotations"]}
    bank_artifact = torch.load(result / "cup_prototype_bank_v1.pt", map_location="cpu", weights_only=True)
    bank = bank_artifact["prototypes"] if isinstance(bank_artifact, dict) else bank_artifact
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dino(device)
    patches, records = {}, []
    for record in manifest["records"]:
        if record["source_split"] != "train":
            raise RuntimeError("Non-train record found")
        image = cv2.imread(str(result / record["image_path"]))
        key = (record["episode_index"], record["frame_index"])
        patches[key] = dino_patches(model, image, device)
        records.append(record)
    check = [r for r in records if r["subset"] == "calibration_check"]
    all_metrics = {}
    tracking_payload = {}
    figure_root = result / "figures" / "temperature_overlays"
    for tau in TAUS:
        rows = []
        for record in check:
            key = (record["episode_index"], record["frame_index"])
            row = measure(patches[key], bank.to(device), boxes[key], tau)
            row.update({"episode_index": key[0], "frame_index": key[1]})
            rows.append(row)
        out_dir = figure_root / f"tau_{tau:.2f}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for record, row in zip(check, rows):
            if record["episode_index"] in (2, 25, 50):
                key = (record["episode_index"], record["frame_index"])
                overlay(cv2.imread(str(result / record["image_path"])), row, boxes[key], out_dir / f"ep{key[0]:03d}.png")
        x_pred = [r["soft_center_xy_pixels"][0] for r in rows]
        y_pred = [r["soft_center_xy_pixels"][1] for r in rows]
        x_true = [r["target_center_xy_pixels"][0] for r in rows]
        y_true = [r["target_center_xy_pixels"][1] for r in rows]
        stats = summary(rows)
        stats.update({
            "tau": tau,
            "pearson_x": corr(x_pred, x_true), "pearson_y": corr(y_pred, y_true),
            "spearman_x": corr(rank(x_pred), rank(x_true)), "spearman_y": corr(rank(y_pred), rank(y_true)),
            "predicted_span_xy": [float(max(x_pred) - min(x_pred)), float(max(y_pred) - min(y_pred))],
            "target_span_xy": [float(max(x_true) - min(x_true)), float(max(y_true) - min(y_true))],
        })
        all_metrics[str(tau)] = stats
        tracking_payload[str(tau)] = [
            {"episode_index": r["episode_index"], "frame_index": r["frame_index"],
             "predicted": r["soft_center_xy_pixels"], "target": r["target_center_xy_pixels"]}
            for r in rows
        ]
    (result / "metrics.json").write_text(json.dumps({"device": str(device), "taus": all_metrics}, indent=2), encoding="utf-8")
    (result / "tracking.json").write_text(json.dumps(tracking_payload, indent=2), encoding="utf-8")
    plot = np.full((620, 900, 3), 255, dtype=np.uint8)
    colors = {"1.0": (130, 130, 130), "0.2": (180, 110, 40), "0.1": (30, 120, 210), "0.05": (40, 170, 70), "0.02": (160, 70, 160)}
    for tau, rows in tracking_payload.items():
        color = colors[tau]
        for item in rows:
            tx, ty = item["target"]; px, py = item["predicted"]
            tx, ty = int(tx * 1.2), int(ty * 1.2)
            px, py = int(px * 1.2), int(py * 1.2)
            cv2.circle(plot, (tx, ty), 4, color, -1)
            cv2.drawMarker(plot, (px, py), color, cv2.MARKER_CROSS, 8, 1)
    cv2.putText(plot, "filled=target, cross=predicted; color=tau", (15, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(result / "figures" / "soft_center_tracking.png"), plot)
    (result / "validation.json").write_text(json.dumps({"experiment": "P0_1_CUP_SPATIAL_POOLING_FIX", "dev_used": False, "final_test_used": False, "taus": list(TAUS)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    main(parser.parse_args())
