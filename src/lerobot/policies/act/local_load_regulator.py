"""F2.3 frozen local MLP adapter. No robot I/O; integration needs F3 safety gates."""
from pathlib import Path

import torch
from torch import nn


class LocalLoadRegulator(nn.Module):
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        super().__init__()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["variant"] != "C1-load":
            raise ValueError("F3 requires the accepted C1-load checkpoint.")
        self.bound = saved["config"]["delta_max"]
        self.normalization = saved["normalization"]
        self.input = nn.Linear(3, 32)
        self.hidden = nn.Linear(32, 32)
        self.output = nn.Linear(32, 1)
        self.load_state_dict(saved["model"], strict=True)
        self.to(device).eval().requires_grad_(False)

    def forward(self, raw_load, raw_position, nominal_gripper):
        norm = self.normalization
        load = (raw_load - norm["load"]["mean"]) / norm["load"]["std"]
        pos = (raw_position - norm["position"]["mean"]) / norm["position"]["std"]
        nominal = (nominal_gripper - norm["nominal"]["mean"]) / norm["nominal"]["std"]
        features = torch.stack([load, pos, nominal], dim=-1)
        if not torch.isfinite(features).all():
            raise ValueError("Nonfinite local inputs; stop rather than impute.")
        value = torch.nn.functional.gelu(self.input(features))
        value = torch.nn.functional.gelu(self.hidden(value))
        return self.bound * torch.tanh(self.output(value)).squeeze(-1)

    @torch.inference_mode()
    def apply(self, native_action, raw_load=None, raw_position=None, mode="disabled"):
        """Inputs [...,6] native actions, gripper last. Returns action and telemetry.

        Call after global postprocessing/ensembling and before unchanged robot safety
        limiting. Raw load must never be sent through the 6D global policy.
        This function does not execute an action or authorize regulated trials.
        """
        if mode not in ("disabled", "shadow", "regulated"):
            raise ValueError("Unknown regulator mode.")
        if native_action.shape[-1] != 6 or not torch.isfinite(native_action).all():
            raise ValueError("Expected finite native actions with gripper last.")
        if mode == "disabled":
            return native_action, None
        nominal = native_action[..., 5]
        if raw_load is None or raw_position is None:
            raise ValueError("Shadow/regulated mode requires same-t raw load and position.")
        if raw_load.shape != nominal.shape or raw_position.shape != nominal.shape:
            raise ValueError("Input shapes must match gripper command shape.")
        delta = self(raw_load, raw_position, nominal)
        if not torch.isfinite(delta).all():
            raise ValueError("Nonfinite local output: stop.")
        clipped = delta.clamp(-self.bound, self.bound)
        hypothetical = native_action.clone()
        hypothetical[..., 5] = nominal + clipped
        if not torch.equal(hypothetical[..., :5], native_action[..., :5]):
            raise RuntimeError("INVALID_LOCAL_ISOLATION")
        executed = hypothetical if mode == "regulated" else native_action
        return executed, dict(predicted_delta=delta, clipped_delta=clipped,
                              nominal_gripper=nominal, hypothetical_gripper=hypothetical[..., 5])
