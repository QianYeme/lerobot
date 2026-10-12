#!/usr/bin/env python
"""Run the frozen M0 OC-ACT engineering gates. No training or dev/test evaluation."""

import argparse
import json
import tempfile
import time
from pathlib import Path

import cv2
import pyarrow.parquet as pq
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_STATE


TOP = "observation.images.top"
WRIST = "observation.images.gripper"


def config(*, oc: bool, prototype_path: Path | None, gate: float, device: str) -> ACTConfig:
    return ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            WRIST: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            TOP: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        chunk_size=4,
        n_action_steps=4,
        use_vae=False,
        dim_model=128,
        n_heads=8,
        dim_feedforward=512,
        n_encoder_layers=2,
        n_decoder_layers=1,
        pretrained_backbone_weights=None,
        model_type="oc_act" if oc else "act",
        object_conditioning_enabled=oc,
        object_conditioning_top_camera_key=TOP,
        object_conditioning_prototype_path=str(prototype_path) if prototype_path else None,
        object_conditioning_num_prototypes=4,
        object_conditioning_num_object_tokens=5,
        object_conditioning_adapter_dim=128,
        object_conditioning_gate_init=gate,
        object_conditioning_smoke_mode=oc and prototype_path is None,
        device=device,
    )


def first_video_frame(path: Path) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot decode first frame from {path}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255)


def real_batch(dataset: Path, device: str) -> dict[str, torch.Tensor]:
    table = pq.read_table(dataset / "data/chunk-000/file-000.parquet").slice(0, 4).to_pydict()
    state = torch.tensor(table[OBS_STATE][0], dtype=torch.float32).unsqueeze(0)
    actions = torch.tensor(table[ACTION], dtype=torch.float32).unsqueeze(0)
    top = first_video_frame(dataset / "videos/observation.images.top/chunk-000/file-000.mp4").unsqueeze(0)
    wrist = first_video_frame(
        dataset / "videos/observation.images.gripper/chunk-000/file-000.mp4"
    ).unsqueeze(0)
    return {
        OBS_STATE: state.to(device),
        TOP: top.to(device),
        WRIST: wrist.to(device),
        ACTION: actions.to(device),
        "action_is_pad": torch.zeros(1, 4, dtype=torch.bool, device=device),
    }


def parameter_counts(policy: ACTPolicy) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in policy.parameters())
    trainable = sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def timed_forward(policy: ACTPolicy, batch: dict[str, torch.Tensor], repeats: int = 5) -> float:
    policy.eval()
    with torch.no_grad():
        for _ in range(2):
            policy.predict_action_chunk(batch)
        if batch[OBS_STATE].is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            policy.predict_action_chunk(batch)
        if batch[OBS_STATE].is_cuda:
            torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repeats


def norm(parameters) -> float:
    values = [parameter.grad.detach().norm() for parameter in parameters if parameter.grad is not None]
    return torch.stack(values).norm().item() if values else 0.0


def run(args):
    result_dir = args.result_dir
    figures_dir = result_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(20261011)

    prototype_path = result_dir / "smoke_prototype_bank_NOT_FOR_FORMAL_TRAINING.pt"
    torch.save(
        {
            "prototypes": torch.randn(4, 384),
            "provenance": "NOT_FOR_FORMAL_TRAINING: deterministic M0 engineering smoke only",
        },
        prototype_path,
    )
    batch = real_batch(args.dataset, device)

    baseline = ACTPolicy(config(oc=False, prototype_path=None, gate=0.0, device=device)).to(device)
    oc = ACTPolicy(config(oc=True, prototype_path=prototype_path, gate=0.0, device=device)).to(device)
    missing, unexpected = oc.load_state_dict(baseline.state_dict(), strict=False)
    allowed_missing = [name for name in missing if ".object_conditioning." in name]
    if len(allowed_missing) != len(missing) or unexpected:
        raise RuntimeError(
            f"Unexpected baseline transfer mismatch: missing={missing}, unexpected={unexpected}"
        )

    baseline.eval()
    oc.eval()
    with torch.no_grad():
        baseline_output = baseline.predict_action_chunk(batch)
        zero_gate_output = oc.predict_action_chunk(batch)
    equivalence_max_abs = (baseline_output - zero_gate_output).abs().max().item()

    baseline.train()
    baseline.zero_grad(set_to_none=True)
    baseline_loss, baseline_loss_dict = baseline(batch)
    baseline_loss.backward()

    oc.train()
    oc.model.object_conditioning.gate.data.fill_(0.2)
    oc.zero_grad(set_to_none=True)
    loss, loss_dict = oc(batch)
    loss.backward()
    object_module = oc.model.object_conditioning
    gradient_norms = {
        "dino": norm(object_module.dino.parameters()),
        "adapter": norm(object_module.adapter.parameters()),
        "cross_attention": norm(object_module.cross_attention.parameters()),
        "gate": float(object_module.gate.grad.abs().item()),
        "act_action_head": norm(oc.model.action_head.parameters()),
    }
    dino_grad_is_none = all(parameter.grad is None for parameter in object_module.dino.parameters())

    object_module.eval()
    with torch.no_grad():
        raw_tokens, similarity_map = object_module.extract_raw_object_tokens(batch[TOP])
        hidden = torch.randn(20, 1, oc.config.dim_model, device=device)
        normal = object_module(hidden, batch[TOP], object_module.adapt_object_tokens(raw_tokens))
        zero = object_module(
            hidden, batch[TOP], object_module.adapt_object_tokens(torch.zeros_like(raw_tokens))
        )
        shuffled = object_module(
            hidden, batch[TOP], object_module.adapt_object_tokens(raw_tokens.flip(1))
        )
        shifted = raw_tokens.clone()
        shifted[:, :, -3:-1] = shifted[:, :, -3:-1].roll(1, dims=1)
        shifted = object_module(hidden, batch[TOP], object_module.adapt_object_tokens(shifted))
    sensitivity = {
        "zero_max_abs": (normal - zero).abs().max().item(),
        "shuffle_max_abs": (normal - shuffled).abs().max().item(),
        "spatial_shift_max_abs": (normal - shifted).abs().max().item(),
    }

    with tempfile.TemporaryDirectory() as temporary_directory:
        baseline_checkpoint = Path(temporary_directory) / "baseline_checkpoint"
        baseline.save_pretrained(baseline_checkpoint)
        baseline_checkpoint_size = sum(
            path.stat().st_size for path in baseline_checkpoint.rglob("*") if path.is_file()
        )
        baseline_reloaded = ACTPolicy.from_pretrained(baseline_checkpoint).to(device).eval()
        baseline_roundtrip_max_abs = max(
            (left.detach().cpu() - right.detach().cpu()).abs().max().item()
            for left, right in zip(
                baseline.state_dict().values(), baseline_reloaded.state_dict().values(), strict=True
            )
        )
        checkpoint = Path(temporary_directory) / "checkpoint"
        oc.save_pretrained(checkpoint)
        checkpoint_size = sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file())
        reloaded = ACTPolicy.from_pretrained(checkpoint).to(device).eval()
        roundtrip_max_abs = max(
            (left.detach().cpu() - right.detach().cpu()).abs().max().item()
            for left, right in zip(oc.state_dict().values(), reloaded.state_dict().values(), strict=True)
            if isinstance(left, torch.Tensor)
        )

    resources = {}
    for name, policy in (("act", baseline), ("oc_act", oc)):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        latency = timed_forward(policy, batch)
        resources[name] = {
            **parameter_counts(policy),
            "parameter_bytes": sum(
                parameter.numel() * parameter.element_size() for parameter in policy.parameters()
            ),
            "forward_latency_ms_batch1": latency,
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        }
    resources["oc_act"]["checkpoint_size_bytes"] = checkpoint_size
    resources["act"]["checkpoint_size_bytes"] = baseline_checkpoint_size

    heatmap = similarity_map[0].detach().float().cpu().numpy()
    heatmap = ((heatmap - heatmap.min()) / max(float(heatmap.max() - heatmap.min()), 1e-8) * 255).astype(
        "uint8"
    )
    heatmap = cv2.resize(heatmap, (640, 480), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(
        str(figures_dir / "object_similarity_smoke_NOT_FOR_FORMAL_TRAINING.png"),
        cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS),
    )

    shapes = dict(object_module.last_debug_shapes)
    shapes.update({"state": list(batch[OBS_STATE].shape), "action_chunk": list(zero_gate_output.shape)})
    gates = {
        "M0-G0": bool(torch.isfinite(baseline_output).all()),
        "M0-G1": hasattr(oc.model, "object_conditioning"),
        "M0-G2": shapes["object_tokens"][1:] == (5, 387) and shapes["action_chunk"] == [1, 4, 6],
        "M0-G3": dino_grad_is_none and gradient_norms["dino"] == 0.0,
        "M0-G4": all(
            gradient_norms[key] > 0
            for key in ("adapter", "cross_attention", "gate", "act_action_head")
        ),
        "M0-G5": equivalence_max_abs == 0.0,
        "M0-G6": all(value > 0 for value in sensitivity.values()),
        "M0-G7": roundtrip_max_abs == 0.0,
        "M0-G8": bool(torch.isfinite(baseline_loss)) and baseline_roundtrip_max_abs == 0.0,
        "M0-G9": all(resources[name]["total"] > 0 for name in ("act", "oc_act")),
    }
    validation = {
        "experiment": "M0_OC_ACT_ARCHITECTURE_BUILD",
        "device": device,
        "dataset_smoke": str(args.dataset),
        "formal_training_performed": False,
        "dev_or_test_read": False,
        "real_robot_performed": False,
        "gates": gates,
        "architecture_ready": all(gates.values()),
    }
    metrics = {
        "shapes": shapes,
        "gradient_norms": gradient_norms,
        "dino_grad_is_none": dino_grad_is_none,
        "gate_zero_equivalence_max_abs": equivalence_max_abs,
        "object_path_sensitivity": sensitivity,
        "checkpoint_roundtrip_max_abs": roundtrip_max_abs,
        "baseline_checkpoint_roundtrip_max_abs": baseline_roundtrip_max_abs,
        "baseline_loss_smoke": {"total": baseline_loss.item(), **baseline_loss_dict},
        "loss_smoke": {"total": loss.item(), **loss_dict},
        "resources": resources,
        "prototype_provenance": object_module.prototype_bank.provenance,
    }
    (result_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"validation": validation, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    run(parser.parse_args())
