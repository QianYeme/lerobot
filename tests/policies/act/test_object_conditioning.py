from pathlib import Path

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_STATE


TOP = "observation.images.top"
WRIST = "observation.images.wrist"


class FakeDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 384, kernel_size=14, stride=14, bias=False)

    def forward_features(self, image):
        patches = self.proj(image).flatten(2).transpose(1, 2)
        return {"x_norm_patchtokens": patches}


@pytest.fixture(autouse=True)
def fake_dino_download(monkeypatch):
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: FakeDINO())


def make_config(*, oc: bool, prototype_path: Path | None = None, gate: float = 0.0) -> ACTConfig:
    return ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            TOP: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            WRIST: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        chunk_size=4,
        n_action_steps=4,
        use_vae=False,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        pretrained_backbone_weights=None,
        model_type="oc_act" if oc else "act",
        object_conditioning_enabled=oc,
        object_conditioning_prototype_path=str(prototype_path) if prototype_path else None,
        object_conditioning_num_prototypes=4,
        object_conditioning_adapter_dim=16,
        object_conditioning_gate_init=gate,
        object_conditioning_smoke_mode=oc and prototype_path is None,
        device="cpu",
    )


def make_batch(batch_size=2):
    return {
        OBS_STATE: torch.randn(batch_size, 8),
        TOP: torch.rand(batch_size, 3, 64, 64),
        WRIST: torch.rand(batch_size, 3, 64, 64),
        ACTION: torch.randn(batch_size, 4, 6),
        "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
    }


def copy_baseline_weights(baseline: ACTPolicy, oc_policy: ACTPolicy):
    oc_policy.load_state_dict(baseline.state_dict(), strict=False)


def test_baseline_has_no_object_conditioning_parameters():
    policy = ACTPolicy(make_config(oc=False))
    assert not hasattr(policy.model, "object_conditioning")
    loss, _ = policy(make_batch())
    loss.backward()
    assert torch.isfinite(loss)


def test_oc_shapes_freeze_and_gradients():
    policy = ACTPolicy(make_config(oc=True, gate=0.2))
    batch = make_batch()
    loss, _ = policy(batch)
    loss.backward()

    oc = policy.model.object_conditioning
    assert oc.last_debug_shapes == {
        "top_rgb": (2, 3, 64, 64),
        "dino_patches": (2, 256, 384),
        "similarity_map": (2, 16, 16),
        "object_tokens": (2, 5, 387),
        "adapter_output": (2, 5, 32),
        "cross_attention_output": (10, 2, 32),
        "act_encoder_input": (10, 2, 32),
    }
    assert all(not parameter.requires_grad for parameter in oc.dino.parameters())
    assert all(parameter.grad is None for parameter in oc.dino.parameters())
    assert oc.gate.grad is not None and oc.gate.grad.abs().item() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in oc.adapter.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in oc.cross_attention.parameters()
    )
    assert policy.model.action_head.weight.grad.abs().sum() > 0


def test_zero_gate_matches_baseline_and_nonzero_gate_is_object_sensitive():
    torch.manual_seed(7)
    baseline = ACTPolicy(make_config(oc=False)).eval()
    torch.manual_seed(8)
    oc_policy = ACTPolicy(make_config(oc=True, gate=0.0)).eval()
    copy_baseline_weights(baseline, oc_policy)
    batch = make_batch()

    with torch.no_grad():
        baseline_actions = baseline.predict_action_chunk(batch)
        oc_actions = oc_policy.predict_action_chunk(batch)
    torch.testing.assert_close(oc_actions, baseline_actions, rtol=0, atol=0)

    oc = oc_policy.model.object_conditioning
    act_tokens = torch.randn(10, 2, 32)
    raw_tokens = torch.randn(2, 5, 387)
    with torch.no_grad():
        oc.gate.fill_(0.2)
    normal_tokens = oc.adapt_object_tokens(raw_tokens)
    normal = oc(act_tokens, batch[TOP], normal_tokens)
    zero = oc(act_tokens, batch[TOP], oc.adapt_object_tokens(torch.zeros_like(raw_tokens)))
    shuffled_tokens = oc.adapt_object_tokens(raw_tokens.flip(1))
    normal = oc(act_tokens, batch[TOP], normal_tokens)
    shuffled = oc(act_tokens, batch[TOP], shuffled_tokens)
    assert not torch.equal(normal, zero)
    spatially_changed = raw_tokens.clone()
    spatially_changed[:, :, -3:-1] = spatially_changed[:, :, -3:-1].roll(1, dims=1)
    shifted = oc(act_tokens, batch[TOP], oc.adapt_object_tokens(spatially_changed))
    assert not torch.equal(normal, shifted)
    assert not torch.equal(normal, shuffled)


def test_checkpoint_roundtrip_restores_oc_state(tmp_path):
    prototype_path = tmp_path / "smoke_prototypes.pt"
    torch.save(
        {
            "prototypes": torch.randn(4, 384),
            "provenance": "NOT_FOR_FORMAL_TRAINING: unit-test prototype bank",
        },
        prototype_path,
    )
    policy = ACTPolicy(make_config(oc=True, prototype_path=prototype_path, gate=0.2)).eval()
    save_path = tmp_path / "checkpoint"
    policy.save_pretrained(save_path)
    loaded = ACTPolicy.from_pretrained(save_path).eval()
    assert loaded.config.model_type == "oc_act"
    assert loaded.model.object_conditioning.prototype_bank.provenance.startswith("NOT_FOR_FORMAL_TRAINING")
    torch.testing.assert_close(policy.state_dict(), loaded.state_dict(), rtol=0, atol=0)
