"""Verify full-size 8-state MASK training and run a finite train-only segmentation fit."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act_det.modeling_act_det import ACTDetPolicy
from lerobot.policies.act_det.detection.mask_decoder import mask_supervision_loss
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.offline_eval_act_det import collate


def segmentation_metrics(prediction, target):
    pred = prediction > 0.5
    gt = target > 0.5
    intersection = (pred & gt).sum(dim=(1, 2, 3)).float()
    union = (pred | gt).sum(dim=(1, 2, 3)).float()
    total = pred.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3))
    return {"l1": float(F.l1_loss(prediction, target)),
            "iou": float(((intersection + 1e-6) / (union + 1e-6)).mean()),
            "dice": float(((2 * intersection + 1e-6) / (total + 1e-6)).mean()),
            "predicted_foreground_fraction": float(pred.float().mean()),
            "label_foreground_fraction": float(gt.float().mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--loss-type", choices=["l1", "bce_dice"], default="l1")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    torch.manual_seed(1000)
    torch.set_num_threads(4)
    cfg = PreTrainedConfig.from_pretrained(args.reference_checkpoint)
    cfg.use_mask_guidance = True
    cfg.mask_dir = str(args.mask_dir)
    cfg.mask_weight = 0.1
    cfg.mask_loss_type = args.loss_type
    cfg.mask_cache_episodes = 1
    cfg.mask_feature_inject = False
    cfg.fcos_feature_inject = False
    cfg.annotation_dir = str(args.dataset_root / "annotations")
    cfg.aug_enable = False
    cfg.device = "cuda"
    policy = ACTDetPolicy(cfg).cuda()
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(args.reference_checkpoint))
    dataset = LeRobotDataset("QYyyyyyyy/formal1_C50_nomaster", root=args.dataset_root, episodes=[0],
                             delta_timestamps={"action": [i / 30 for i in range(cfg.chunk_size)]}, video_backend="pyav")
    indices = np.linspace(0, len(dataset) - 1, 32, dtype=int).tolist()
    examples = [dataset[i] for i in indices]
    batch = collate(examples[:2])
    frames = batch["frame_index"]
    batch = pre(batch)
    batch["frame_index"] = frames.cuda()
    policy.train()
    loss, logs = policy(batch)
    mask_loss = policy.model.get_mask_loss()["mask_loss"]
    mask_loss.backward()
    gradient_norms = {}
    for name in ("mask_decoder", "fpn", "backbone"):
        gradients = [p.grad for p in getattr(policy.model, name).parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise AssertionError(f"Invalid MASK gradient: {name}")
        gradient_norms[name] = sum(float(g.square().sum()) for g in gradients) ** 0.5
        if gradient_norms[name] == 0:
            raise AssertionError(f"Zero MASK gradient: {name}")
    if not any(k.startswith("det_") for k in logs) or logs["mask_coverage"] != 1:
        raise AssertionError(f"Missing detection or MASK supervision: {logs}")
    policy.zero_grad(set_to_none=True)
    policy.eval()
    with torch.inference_mode():
        reference = policy.predict_action_chunk(batch)
        loader = policy.model.mask_loader
        policy.model.mask_loader = None
        without_labels = policy.predict_action_chunk(batch)
        policy.model.mask_loader = loader
        torch.testing.assert_close(reference, without_labels)
        if not torch.isfinite(post(reference[:, 0])).all():
            raise AssertionError("Invalid postprocessed action")
    report = {"full_policy_gradient_check": "passed", "state_dim": 8, "action_dim": 6,
              "image_shape": [480, 640], "gradient_norms": gradient_norms,
              "inference_without_masks": "passed", "loss_components": logs,
              "fit": "train-only sanity check; random policy initialization; not a task-trained checkpoint",
              "episode": 0, "frames": indices, "steps": args.steps, "loss_type": args.loss_type,
              "fit_lr": {"backbone": 1e-4, "fpn": 1e-4, "decoder": 1e-3}, "curve": []}
    print(json.dumps({"gradient_check": "passed", "gradient_norms": gradient_norms}), flush=True)
    del loss, mask_loss, batch, reference, without_labels
    policy.model._mask_loss = None
    policy.model._det_loss = None
    images = []
    labels = []
    with torch.no_grad():
        for example in examples:
            processed = pre(collate([example]))
            images.append(processed["observation.images.top"][0])
            mask = loader.get_mask("observation.images.top", 0, int(example["frame_index"]))
            if mask is None:
                raise ValueError("Sanity-fit labels must be valid")
            labels.append(torch.from_numpy(mask.copy()).float()[None])
    images = torch.stack(images)
    target = torch.stack(labels).cuda()
    model = policy.model
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-4},
        {"params": model.fpn.parameters(), "lr": 1e-4},
        {"params": model.mask_decoder.parameters(), "lr": 1e-3},
    ])

    def predict(x):
        features = model.backbone(x)
        return model.mask_decoder(*model.fpn([features["f2"], features["f3"], features["f4"]]),
                                  return_logits=args.loss_type == "bce_dice")

    def evaluate(step):
        model.eval()
        with torch.inference_mode():
            predictions = torch.cat([predict(images[start:start + 4]) for start in range(0, len(images), 4)])
            if args.loss_type == "bce_dice":
                predictions = predictions.sigmoid()
            entry = {"step": step, **segmentation_metrics(predictions, target)}
        report["curve"].append(entry)
        print(json.dumps(entry), flush=True)
        (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return predictions

    evaluate(0)
    generator = torch.Generator().manual_seed(1000)
    for step in range(1, args.steps + 1):
        model.train()
        selected = torch.randperm(len(images), generator=generator)[:4].cuda()
        optimizer.zero_grad(set_to_none=True)
        prediction = predict(images[selected])
        fit_loss = mask_supervision_loss(prediction, target[selected], args.loss_type)
        fit_loss.backward()
        optimizer.step()
        if step % 50 == 0:
            evaluate(step)
    predictions = evaluate(args.steps) if report["curve"][-1]["step"] != args.steps else None
    if predictions is None:
        model.eval()
        with torch.inference_mode():
            predictions = torch.cat([predict(images[start:start + 4]) for start in range(0, len(images), 4)])
            if args.loss_type == "bce_dice":
                predictions = predictions.sigmoid()
    sheet = Image.new("RGB", (1280, 3 * 510), "white")
    for panel, i in enumerate((0, 6, 12, 18, 24, 31)):
        rgb = (examples[i]["observation.images.top"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        foreground = predictions[i, 0].cpu().numpy() > 0.5
        rgb[foreground] = (0.6 * rgb[foreground] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
        x, y = panel % 2 * 640, panel // 2 * 510
        sheet.paste(Image.fromarray(rgb), (x, y + 30))
        ImageDraw.Draw(sheet).text((x + 8, y + 8), f"fit prediction: frame={indices[i]}", fill="black")
    sheet.save(args.output / "fit_predictions.jpg", quality=95)
    final = report["curve"][-1]
    report["fit_status"] = "passed" if final["iou"] >= 0.5 and final["dice"] > report["curve"][0]["dice"] else "foreground_learning_insufficient"
    report["formal_training_ready"] = False
    report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "check.done").write_text("check_complete; formal_training_not_ready\n", encoding="utf-8")
    print(json.dumps({"fit_status": report["fit_status"], "final": final}), flush=True)


if __name__ == "__main__":
    main()
