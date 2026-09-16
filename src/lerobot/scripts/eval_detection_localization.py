#!/usr/bin/env python

"""Evaluate ACTDet FCOS localization quality on annotated LeRobot frames."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou, nms

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.offline_eval_act_det import POLICY_CLASSES, collate, parse_episodes
from lerobot.utils.constants import OBS_IMAGES


def decode_fcos(
    cls_logits: list[torch.Tensor],
    reg_preds: list[torch.Tensor],
    ctr_preds: list[torch.Tensor],
    strides: list[int],
    image_size: tuple[int, int],
    score_floor: float,
    nms_iou: float,
    pre_nms_topk: int,
    max_detections: int,
) -> list[dict[str, torch.Tensor]]:
    """Decode single-class FCOS outputs into boxes and confidence scores."""
    height, width = image_size
    batch_size = cls_logits[0].shape[0]
    decoded = []

    for batch_idx in range(batch_size):
        image_boxes = []
        image_scores = []
        for cls_level, reg_level, ctr_level, stride in zip(
            cls_logits, reg_preds, ctr_preds, strides, strict=True
        ):
            if cls_level.shape[1] != 1:
                raise ValueError("This evaluator currently supports one FCOS class")

            _, _, level_height, level_width = cls_level.shape
            scores = torch.sqrt(
                cls_level[batch_idx, 0].sigmoid() * ctr_level[batch_idx, 0].sigmoid()
            ).flatten()
            candidate_count = min(pre_nms_topk, scores.numel())
            scores, flat_indices = scores.topk(candidate_count)
            keep = scores >= score_floor
            scores = scores[keep]
            flat_indices = flat_indices[keep]
            if scores.numel() == 0:
                continue

            rows = torch.div(flat_indices, level_width, rounding_mode="floor")
            cols = flat_indices % level_width
            regression = reg_level[batch_idx].permute(1, 2, 0).reshape(-1, 4)[flat_indices]
            center_x = (cols.to(regression.dtype) + 0.5) * stride
            center_y = (rows.to(regression.dtype) + 0.5) * stride
            boxes = torch.stack(
                [
                    center_x - regression[:, 0] * stride,
                    center_y - regression[:, 1] * stride,
                    center_x + regression[:, 2] * stride,
                    center_y + regression[:, 3] * stride,
                ],
                dim=1,
            )
            image_boxes.append(boxes)
            image_scores.append(scores)

        if not image_boxes:
            decoded.append(
                {
                    "boxes": torch.empty((0, 4)),
                    "scores": torch.empty((0,)),
                }
            )
            continue

        boxes = torch.cat(image_boxes)
        scores = torch.cat(image_scores)
        boxes[:, 0::2].clamp_(0, width)
        boxes[:, 1::2].clamp_(0, height)
        valid = (
            torch.isfinite(boxes).all(dim=1)
            & torch.isfinite(scores)
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        boxes = boxes[valid]
        scores = scores[valid]
        keep = nms(boxes, scores, nms_iou)[:max_detections]
        decoded.append({"boxes": boxes[keep].cpu(), "scores": scores[keep].cpu()})

    return decoded


def match_predictions(
    predictions: dict[int, dict[str, torch.Tensor]],
    ground_truth: dict[int, torch.Tensor],
    iou_threshold: float,
    score_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    records = []
    for image_id, prediction in predictions.items():
        for score, box in zip(prediction["scores"], prediction["boxes"], strict=True):
            if float(score) >= score_threshold:
                records.append((float(score), image_id, box))
    records.sort(key=lambda item: item[0], reverse=True)

    matched = {image_id: set() for image_id in ground_truth}
    true_positive = []
    false_positive = []
    for _, image_id, box in records:
        gt_boxes = ground_truth[image_id]
        if gt_boxes.numel() == 0:
            true_positive.append(0.0)
            false_positive.append(1.0)
            continue
        ious = box_iou(box.unsqueeze(0), gt_boxes).squeeze(0)
        best_iou, best_idx = ious.max(dim=0)
        gt_idx = int(best_idx)
        if float(best_iou) >= iou_threshold and gt_idx not in matched[image_id]:
            matched[image_id].add(gt_idx)
            true_positive.append(1.0)
            false_positive.append(0.0)
        else:
            true_positive.append(0.0)
            false_positive.append(1.0)

    total_ground_truth = sum(len(boxes) for boxes in ground_truth.values())
    return torch.tensor(true_positive), torch.tensor(false_positive), total_ground_truth


def average_precision(
    predictions: dict[int, dict[str, torch.Tensor]],
    ground_truth: dict[int, torch.Tensor],
    iou_threshold: float,
    score_floor: float,
) -> float:
    true_positive, false_positive, total_gt = match_predictions(
        predictions, ground_truth, iou_threshold, score_floor
    )
    if total_gt == 0 or true_positive.numel() == 0:
        return 0.0
    cumulative_tp = true_positive.cumsum(0)
    cumulative_fp = false_positive.cumsum(0)
    recall = cumulative_tp / total_gt
    precision = cumulative_tp / (cumulative_tp + cumulative_fp).clamp_min(1)
    interpolated = []
    for recall_level in torch.linspace(0, 1, 101):
        eligible = precision[recall >= recall_level]
        interpolated.append(float(eligible.max()) if eligible.numel() else 0.0)
    return sum(interpolated) / len(interpolated)


def operating_point(
    predictions: dict[int, dict[str, torch.Tensor]],
    ground_truth: dict[int, torch.Tensor],
    iou_threshold: float,
    score_threshold: float,
) -> dict[str, float | int]:
    true_positive, false_positive, total_gt = match_predictions(
        predictions, ground_truth, iou_threshold, score_threshold
    )
    tp = int(true_positive.sum())
    fp = int(false_positive.sum())
    fn = total_gt - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / total_gt if total_gt else 0.0
    return {
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def top1_metrics(
    predictions: dict[int, dict[str, torch.Tensor]],
    ground_truth: dict[int, torch.Tensor],
    image_size: tuple[int, int],
) -> dict[str, float | int]:
    ious = []
    center_errors = []
    scores = []
    height, width = image_size
    diagonal = (height**2 + width**2) ** 0.5
    missing = 0
    for image_id, gt_boxes in ground_truth.items():
        if gt_boxes.numel() == 0:
            continue
        prediction = predictions[image_id]
        if prediction["scores"].numel() == 0:
            missing += 1
            ious.append(0.0)
            center_errors.append(1.0)
            scores.append(0.0)
            continue
        box = prediction["boxes"][0]
        scores.append(float(prediction["scores"][0]))
        overlaps = box_iou(box.unsqueeze(0), gt_boxes).squeeze(0)
        best_idx = int(overlaps.argmax())
        ious.append(float(overlaps[best_idx]))
        gt_box = gt_boxes[best_idx]
        pred_center = (box[:2] + box[2:]) / 2
        gt_center = (gt_box[:2] + gt_box[2:]) / 2
        center_errors.append(float(torch.linalg.vector_norm(pred_center - gt_center)) / diagonal)

    iou_tensor = torch.tensor(ious)
    return {
        "annotated_frames": len(ious),
        "missing_predictions": missing,
        "mean_iou": float(iou_tensor.mean()) if ious else 0.0,
        "median_iou": float(iou_tensor.median()) if ious else 0.0,
        "recall_iou_50": float((iou_tensor >= 0.5).float().mean()) if ious else 0.0,
        "recall_iou_75": float((iou_tensor >= 0.75).float().mean()) if ious else 0.0,
        "mean_center_error_normalized": sum(center_errors) / len(center_errors) if center_errors else 0.0,
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
    }


def evaluate(args: argparse.Namespace) -> dict:
    checkpoint = Path(args.checkpoint)
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    if cfg.type != "act_det" or not cfg.use_detection:
        raise ValueError(f"Checkpoint must be an ACTDet policy with detection enabled: {checkpoint}")
    if args.camera_key not in cfg.image_features:
        raise ValueError(f"Unknown camera {args.camera_key!r}; available: {list(cfg.image_features)}")
    if not cfg.det_cameras.get(args.camera_key, {}).get("enable", False):
        raise ValueError(f"Detection is disabled for {args.camera_key!r}")

    annotation_dir = Path(args.annotation_dir)
    camera_dir = annotation_dir / args.camera_key.split(".")[-1]
    episodes = parse_episodes(args.episodes)
    missing_annotations = [
        episode for episode in episodes if not (camera_dir / f"episode_{episode:03d}.xml").is_file()
    ]
    if missing_annotations:
        raise FileNotFoundError(f"Missing annotation XML for episodes: {missing_annotations}")

    cfg.annotation_dir = str(annotation_dir)
    policy = POLICY_CLASSES[cfg.type].from_pretrained(checkpoint, config=cfg)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(checkpoint))

    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        episodes=episodes,
        video_backend=args.dataset_video_backend,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=False,
    )

    camera_idx = list(cfg.image_features).index(args.camera_key)
    predictions = {}
    ground_truth = {}
    image_id = 0
    image_size = tuple(metadata.features[args.camera_key]["shape"][:2])
    start = time.perf_counter()

    with torch.no_grad():
        for step, raw_batch in enumerate(dataloader):
            if args.max_batches is not None and step >= args.max_batches:
                break
            episode_indices = [int(value) for value in raw_batch["episode_index"]]
            frame_indices = [int(value) for value in raw_batch["frame_index"]]
            batch = preprocessor(raw_batch)
            images = [batch[key] for key in cfg.image_features]
            image = images[camera_idx]
            backbone_features = policy.model.backbone(image)
            fpn_features = policy.model.fpn(
                [backbone_features["f2"], backbone_features["f3"], backbone_features["f4"]]
            )
            cls_logits, reg_preds, ctr_preds = policy.model.fcos_head(fpn_features)
            batch_predictions = decode_fcos(
                cls_logits,
                reg_preds,
                ctr_preds,
                cfg.fcos_strides,
                image_size,
                args.score_floor,
                args.nms_iou,
                args.pre_nms_topk,
                args.max_detections,
            )

            for episode, frame, prediction in zip(
                episode_indices, frame_indices, batch_predictions, strict=True
            ):
                labels = policy.model.label_loader.get_labels(args.camera_key, episode, frame)
                boxes = labels["bboxes"] if labels else []
                predictions[image_id] = prediction
                ground_truth[image_id] = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
                image_id += 1
            if step % 50 == 0:
                logging.info(
                    "Step %d (%d frames), %.1fs elapsed",
                    step,
                    image_id,
                    time.perf_counter() - start,
                )

    iou_thresholds = [round(0.50 + index * 0.05, 2) for index in range(10)]
    ap_by_iou = {
        f"{threshold:.2f}": average_precision(predictions, ground_truth, threshold, args.score_floor)
        for threshold in iou_thresholds
    }
    result = {
        "checkpoint": str(checkpoint),
        "dataset": args.dataset_repo_id,
        "episodes": episodes,
        "camera_key": args.camera_key,
        "frames": image_id,
        "ground_truth_boxes": sum(len(boxes) for boxes in ground_truth.values()),
        "protocol": {
            "score_floor_for_ap": args.score_floor,
            "nms_iou": args.nms_iou,
            "pre_nms_topk": args.pre_nms_topk,
            "max_detections_per_frame": args.max_detections,
        },
        "metrics": {
            "ap50": ap_by_iou["0.50"],
            "map50_95": sum(ap_by_iou.values()) / len(ap_by_iou),
            "ap_by_iou": ap_by_iou,
            "top1": top1_metrics(predictions, ground_truth, image_size),
            "operating_points_iou50": [
                operating_point(predictions, ground_truth, 0.5, threshold)
                for threshold in args.operating_score_thresholds
            ],
        },
        "elapsed_seconds": time.perf_counter() - start,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", required=True)
    parser.add_argument("--dataset.root", dest="dataset_root", required=True)
    parser.add_argument("--dataset.video_backend", dest="dataset_video_backend", default="pyav")
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--episodes", default="7,11,13,14,16,27,35,38,40,43")
    parser.add_argument("--camera-key", default="observation.images.top")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--score-floor", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.6)
    parser.add_argument("--pre-nms-topk", type=int, default=1000)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument("--operating-score-thresholds", type=float, nargs="+", default=[0.05, 0.25, 0.5])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = evaluate(args)
    print(json.dumps(result["metrics"], indent=2))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
