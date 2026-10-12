#!/usr/bin/env python
"""Build and diagnose a train-only DINOv2-S cup prototype bank for P0."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812


def load_dino(device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
    model.requires_grad_(False).eval().to(device)
    return model


def dino_patches(model, image_bgr, device):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
    mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = tensor.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    with torch.no_grad():
        patches = model.forward_features((tensor - mean) / std)["x_norm_patchtokens"][0]
    return patches


def patch_geometry(box):
    centers = torch.arange(16, dtype=torch.float32) * 14 + 7
    yy, xx = torch.meshgrid(centers, centers, indexing="ij")
    x1, y1, x2, y2 = [value * 224 / scale for value, scale in zip(box, (640, 480, 640, 480))]
    inside = (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = max((x2 - x1) / 2, 1), max((y2 - y1) / 2, 1)
    ellipse = ((xx - center_x) / half_w).square() + ((yy - center_y) / half_h).square() <= 1.0
    selected = inside & ellipse
    return selected.flatten(), inside.flatten(), (xx, yy), (center_x, center_y)


def deterministic_kmedoids(features, count):
    features = F.normalize(features, dim=-1)
    mean = F.normalize(features.mean(0, keepdim=True), dim=-1)
    selected = [int((features @ mean.T).argmax())]
    while len(selected) < count:
        similarity = features @ features[selected].T
        distance = 1 - similarity.max(dim=1).values
        distance[selected] = -1
        selected.append(int(distance.argmax()))
    return features[selected], selected


def soft_center(similarity):
    weights = torch.softmax(similarity, dim=0).reshape(16, 16)
    coordinates = torch.linspace(-1, 1, 16, device=similarity.device)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    return float((weights * xx).sum()), float((weights * yy).sum()), weights


def normalized_to_pixels(x, y):
    return (x + 1) * 0.5 * 640, (y + 1) * 0.5 * 480


def metric_for_frame(patches, prototypes, box):
    similarity = F.normalize(patches, dim=-1) @ F.normalize(prototypes, dim=-1).T
    similarity = similarity.max(dim=-1).values
    selected, inside, _, target_center_224 = patch_geometry(box)
    x_soft, y_soft, weights = soft_center(similarity)
    pred_x, pred_y = normalized_to_pixels(x_soft, y_soft)
    target_x = target_center_224[0] * 640 / 224
    target_y = target_center_224[1] * 480 / 224
    error = ((pred_x - target_x) ** 2 + (pred_y - target_y) ** 2) ** 0.5
    peak_index = int(similarity.argmax())
    peak_row, peak_col = divmod(peak_index, 16)
    neighbor = inside.reshape(16, 16).clone()
    neighbor = F.max_pool2d(neighbor.float()[None, None], 3, stride=1, padding=1)[0, 0].bool()
    inside_mean = float(similarity[inside].mean())
    outside_mean = float(similarity[~inside].mean())
    corner_mask = torch.zeros(16, 16, dtype=torch.bool, device=similarity.device)
    corner_mask[:2, :2] = corner_mask[:2, -2:] = True
    corner_mask[-2:, :2] = corner_mask[-2:, -2:] = True
    return {
        "soft_center_xy_pixels": [pred_x, pred_y],
        "target_center_xy_pixels": [target_x, target_y],
        "soft_center_error_pixels": error,
        "soft_center_error_normalized_diagonal": error / 800,
        "inside_mean_similarity": inside_mean,
        "outside_mean_similarity": outside_mean,
        "inside_outside_mean_ratio": inside_mean / max(outside_mean, 1e-8),
        "inside_softmax_mass": float(weights.flatten()[inside].sum()),
        "peak_patch_rc": [peak_row, peak_col],
        "peak_in_box": bool(inside[peak_index]),
        "peak_in_box_or_one_patch": bool(neighbor[peak_row, peak_col]),
        "corner_max_similarity": float(similarity[corner_mask.flatten()].max()),
        "cup_max_similarity": float(similarity[inside].max()),
        "similarity": similarity.reshape(16, 16).detach().cpu(),
        "selected_patch_count": int(selected.sum()),
    }


def summarize(rows):
    errors = np.array([row["soft_center_error_pixels"] for row in rows])
    ratios = np.array([row["inside_outside_mean_ratio"] for row in rows])
    masses = np.array([row["inside_softmax_mass"] for row in rows])
    return {
        "count": len(rows),
        "soft_center_error_pixels_median": float(np.median(errors)),
        "soft_center_error_pixels_p90": float(np.quantile(errors, 0.9)),
        "peak_in_box_rate": float(np.mean([row["peak_in_box"] for row in rows])),
        "peak_in_box_or_one_patch_rate": float(
            np.mean([row["peak_in_box_or_one_patch"] for row in rows])
        ),
        "inside_outside_mean_ratio_mean": float(ratios.mean()),
        "inside_softmax_mass_mean": float(masses.mean()),
        "cup_beats_corner_rate": float(
            np.mean([row["cup_max_similarity"] > row["corner_max_similarity"] for row in rows])
        ),
    }


def overlay(image, row, box, output):
    heat = row["similarity"].numpy()
    heat = ((heat - heat.min()) / max(float(heat.max() - heat.min()), 1e-8) * 255).astype(np.uint8)
    heat = cv2.resize(heat, (640, 480), interpolation=cv2.INTER_NEAREST)
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS)
    visual = cv2.addWeighted(image, 0.58, colored, 0.42, 0)
    x1, y1, x2, y2 = box
    cv2.rectangle(visual, (x1, y1), (x2, y2), (0, 255, 0), 2)
    target = tuple(round(value) for value in row["target_center_xy_pixels"])
    predicted = tuple(round(value) for value in row["soft_center_xy_pixels"])
    cv2.drawMarker(visual, target, (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
    cv2.drawMarker(visual, predicted, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
    cv2.line(visual, (320, 0), (320, 479), (255, 255, 255), 1)
    cv2.line(visual, (0, 240), (639, 240), (255, 255, 255), 1)
    cv2.imwrite(str(output), visual)


def token_metrics(patches_by_id, prototypes, records):
    all_tokens = []
    for record in records:
        patches = patches_by_id[(record["episode_index"], record["frame_index"])]
        similarity = (F.normalize(patches, dim=-1) @ prototypes.T).max(dim=-1).values.reshape(16, 16)
        features = patches.reshape(16, 16, 384)
        coords = torch.linspace(-1, 1, 16, device=patches.device)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        regions = [
            (slice(None), slice(None)),
            (slice(0, 8), slice(0, 8)),
            (slice(0, 8), slice(8, 16)),
            (slice(8, 16), slice(0, 8)),
            (slice(8, 16), slice(8, 16)),
        ]
        tokens = []
        for rows, cols in regions:
            sim = similarity[rows, cols].flatten()
            weights = torch.softmax(sim, dim=0)
            feat = features[rows, cols].reshape(-1, 384)
            semantic = weights @ feat
            x = weights @ xx[rows, cols].flatten()
            y = weights @ yy[rows, cols].flatten()
            confidence = weights @ sim
            tokens.append(torch.cat((semantic, x[None], y[None], confidence[None])))
        all_tokens.append(torch.stack(tokens))
    tokens = torch.stack(all_tokens)
    local_cosine = F.cosine_similarity(tokens[:, 1:, :384].unsqueeze(2), tokens[:, 1:, :384].unsqueeze(1), dim=-1)
    off_diagonal = ~torch.eye(4, dtype=torch.bool, device=tokens.device)
    return {
        "shape": list(tokens.shape),
        "finite": bool(torch.isfinite(tokens).all()),
        "x_range": [float(tokens[:, :, -3].min()), float(tokens[:, :, -3].max())],
        "y_range": [float(tokens[:, :, -2].min()), float(tokens[:, :, -2].max())],
        "confidence_range": [float(tokens[:, :, -1].min()), float(tokens[:, :, -1].max())],
        "local_pairwise_cosine_mean": float(local_cosine[:, off_diagonal].mean()),
        "fallback_rule": "softmax over every non-empty 8x8 local patch grid; no empty-token branch",
    }


def main(args):
    result = args.result_dir
    manifest = json.loads((result / "reference_manifest.json").read_text(encoding="utf-8"))
    annotations = json.loads((result / "coarse_box_annotations.json").read_text(encoding="utf-8"))
    if any(row["source_split"] != "train" for row in manifest["records"]):
        raise RuntimeError("Non-train record found.")
    if any(row["review_status"] != "VISUALLY_REVIEWED_PASS_2026-10-11" for row in annotations["annotations"]):
        raise RuntimeError("Every coarse box must pass overlay review before calibration.")
    box_by_id = {
        (row["episode_index"], row["frame_index"]): row["coarse_box_xyxy"]
        for row in annotations["annotations"]
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dino(device)
    patches_by_id = {}
    candidate_features = []
    for record in manifest["records"]:
        image = cv2.imread(str(result / record["image_path"]))
        patches = dino_patches(model, image, device)
        key = (record["episode_index"], record["frame_index"])
        patches_by_id[key] = patches
        if record["subset"] == "reference":
            selected, _, _, _ = patch_geometry(box_by_id[key])
            candidate_features.append(patches[selected.to(device)])
    candidate_features = torch.cat(candidate_features)

    candidate_results = {}
    banks = {}
    for count in (1, 4, 8):
        bank, medoid_indices = deterministic_kmedoids(candidate_features, count)
        banks[count] = bank
        frame_rows = []
        for record in manifest["records"]:
            key = (record["episode_index"], record["frame_index"])
            row = metric_for_frame(patches_by_id[key], bank, box_by_id[key])
            row.update({"episode_index": key[0], "frame_index": key[1], "subset": record["subset"]})
            frame_rows.append(row)
        check_rows = [row for row in frame_rows if row["subset"] == "calibration_check"]
        candidate_results[count] = {
            "prototype_count": count,
            "medoid_indices": medoid_indices,
            "reference": summarize([row for row in frame_rows if row["subset"] == "reference"]),
            "calibration_check": summarize(check_rows),
            "per_frame": frame_rows,
        }

    four = candidate_results[4]["calibration_check"]
    eight = candidate_results[8]["calibration_check"]
    promote_eight = (
        eight["peak_in_box_or_one_patch_rate"] - four["peak_in_box_or_one_patch_rate"] >= 0.15
        or (
            four["soft_center_error_pixels_median"] - eight["soft_center_error_pixels_median"] >= 10
            and eight["peak_in_box_or_one_patch_rate"] >= four["peak_in_box_or_one_patch_rate"]
        )
    )
    selected_count = 8 if promote_eight else 4
    selected_bank = banks[selected_count].detach().cpu()
    bank_path = result / "cup_prototype_bank_v1.pt"
    provenance = {
        "artifact_status": "FORMAL_P0_CANDIDATE",
        "dino_identifier": "dinov2_vits14",
        "feature_dim": 384,
        "prototype_count": selected_count,
        "source_reference_ids": [
            {"episode_index": row["episode_index"], "frame_index": row["frame_index"]}
            for row in manifest["records"]
            if row["subset"] == "reference"
        ],
        "construction_algorithm": "ellipse-filtered coarse-box DINO patches; deterministic farthest-first cosine medoids",
        "similarity_rule": "max cosine similarity across prototypes",
        "selection_rule_frozen_before_results": "default 4; promote to 8 only for >=0.15 neighbor-peak-rate gain or >=10px median-center gain without neighbor-peak regression",
        "selected_count_reason": "8 met pre-frozen promotion rule" if promote_eight else "4 remained default; 8 did not meet pre-frozen promotion rule",
        "train_only": True,
        "dev_used": False,
        "final_test_used": False,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
    }
    torch.save({"prototypes": selected_bank, "provenance": provenance}, bank_path)
    provenance["sha256"] = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    (result / "prototype_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    selected_rows = candidate_results[selected_count]["per_frame"]
    for row in selected_rows:
        record = next(
            item
            for item in manifest["records"]
            if item["episode_index"] == row["episode_index"] and item["frame_index"] == row["frame_index"]
        )
        folder = result / "figures" / (
            "reference_similarity_overlays"
            if row["subset"] == "reference"
            else "train_check_similarity_overlays"
        )
        folder.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(result / record["image_path"]))
        overlay(
            image,
            row,
            box_by_id[(row["episode_index"], row["frame_index"])],
            folder / f"ep{row['episode_index']:03d}_frame{row['frame_index']:03d}.png",
        )

    serializable_candidates = {}
    for count, payload in candidate_results.items():
        serializable = {key: value for key, value in payload.items() if key != "per_frame"}
        serializable["per_frame"] = [
            {key: value for key, value in row.items() if key != "similarity"}
            for row in payload["per_frame"]
        ]
        serializable_candidates[str(count)] = serializable
    check_records = [row for row in manifest["records"] if row["subset"] == "calibration_check"]
    tokens = token_metrics(patches_by_id, F.normalize(banks[selected_count], dim=-1), check_records)
    metrics = {
        "device": str(device),
        "candidate_patch_count": int(candidate_features.shape[0]),
        "candidate_comparison": serializable_candidates,
        "selected_prototype_count": selected_count,
        "selected_check_metrics": candidate_results[selected_count]["calibration_check"],
        "object_tokens": tokens,
    }
    (result / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected_count, "check": metrics["selected_check_metrics"], "tokens": tokens}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    main(parser.parse_args())
