#!/usr/bin/env python

from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.act.configuration_act import ACTConfig


class FrozenDINOv2Small(nn.Module):
    """Frozen pretrained DINOv2-S patch encoder for the top camera only."""

    feature_dim = 384

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        local_weights = Path(torch.hub.get_dir()) / "checkpoints" / f"{model_name}_pretrain.pth"
        local_repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        hub_kwargs = {"source": "local"} if local_repo.exists() else {}
        if local_weights.exists():
            self.model = torch.hub.load(str(local_repo) if local_repo.exists() else "facebookresearch/dinov2", model_name, pretrained=False, **hub_kwargs)
            state = torch.load(local_weights, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state, strict=True)
        else:
            self.model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, image: Tensor) -> tuple[Tensor, tuple[int, int]]:
        # ACT images are [0,1]. DINOv2 expects ImageNet normalization and a patch-aligned input.
        image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
        mean = image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        with torch.no_grad():
            features = self.model.forward_features((image - mean) / std)["x_norm_patchtokens"]
        side = int(features.shape[1] ** 0.5)
        if side * side != features.shape[1]:
            raise RuntimeError(f"DINO patch count {features.shape[1]} is not a square grid.")
        return features, (side, side)


class PrototypeBank(nn.Module):
    def __init__(self, config: ACTConfig, feature_dim: int) -> None:
        super().__init__()
        self.source = config.object_conditioning_prototype_path
        self.smoke_only = self.source is None
        if self.smoke_only:
            generator = torch.Generator().manual_seed(0)
            prototypes = torch.randn(
                config.object_conditioning_num_prototypes, feature_dim, generator=generator
            )
            self.provenance = "NOT_FOR_FORMAL_TRAINING: deterministic synthetic smoke prototypes"
        else:
            payload = torch.load(Path(self.source), map_location="cpu", weights_only=True)
            prototypes = payload["prototypes"] if isinstance(payload, dict) else payload
            self.provenance = (
                str(payload.get("provenance", self.source)) if isinstance(payload, dict) else self.source
            )
        expected = (config.object_conditioning_num_prototypes, feature_dim)
        if tuple(prototypes.shape) != expected:
            raise ValueError(f"Prototype bank must have shape {expected}, got {tuple(prototypes.shape)}.")
        self.register_buffer("prototypes", F.normalize(prototypes.float(), dim=-1))

class ObjectTokenizer(nn.Module):
    """Create one global and four 2x2-local soft object tokens."""

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def forward(
        self, patches: Tensor, grid_shape: tuple[int, int], prototypes: Tensor
    ) -> tuple[Tensor, Tensor]:
        batch_size, num_patches, feature_dim = patches.shape
        height, width = grid_shape
        if num_patches != height * width:
            raise ValueError("Patch tensor and spatial grid disagree.")
        normalized = F.normalize(patches, dim=-1)
        similarity = torch.einsum("bnd,pd->bnp", normalized, prototypes).amax(dim=-1)
        feature_grid = patches.reshape(batch_size, height, width, feature_dim)
        similarity_grid = similarity.reshape(batch_size, height, width)
        y = torch.linspace(-1, 1, height, device=patches.device, dtype=patches.dtype)
        x = torch.linspace(-1, 1, width, device=patches.device, dtype=patches.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        regions = [
            (slice(None), slice(None)),
            (slice(0, height // 2), slice(0, width // 2)),
            (slice(0, height // 2), slice(width // 2, width)),
            (slice(height // 2, height), slice(0, width // 2)),
            (slice(height // 2, height), slice(width // 2, width)),
        ]
        tokens = []
        for rows, cols in regions:
            region_features = feature_grid[:, rows, cols].flatten(1, 2)
            region_similarity = similarity_grid[:, rows, cols].flatten(1)
            weights = torch.softmax(region_similarity / self.temperature, dim=1)
            semantic = torch.einsum("bn,bnd->bd", weights, region_features)
            region_x = xx[rows, cols].flatten()
            region_y = yy[rows, cols].flatten()
            center_x = torch.einsum("bn,n->b", weights, region_x).unsqueeze(-1)
            center_y = torch.einsum("bn,n->b", weights, region_y).unsqueeze(-1)
            confidence = torch.einsum("bn,bn->b", weights, region_similarity).unsqueeze(-1)
            tokens.append(torch.cat((semantic, center_x, center_y, confidence), dim=-1))
        return torch.stack(tokens, dim=1), similarity_grid


class ObjectConditioning(nn.Module):
    def __init__(self, config: ACTConfig) -> None:
        super().__init__()
        self.dino = FrozenDINOv2Small(config.object_conditioning_dino_model)
        self.prototype_bank = PrototypeBank(config, self.dino.feature_dim)
        self.tokenizer = ObjectTokenizer(config.object_conditioning_temperature)
        token_dim = self.dino.feature_dim + 3
        self.adapter = nn.Sequential(
            nn.Linear(token_dim, config.object_conditioning_adapter_dim),
            nn.LayerNorm(config.object_conditioning_adapter_dim),
            nn.GELU(),
            nn.Linear(config.object_conditioning_adapter_dim, config.dim_model),
        )
        self.token_slot_embedding = nn.Embedding(
            config.object_conditioning_num_object_tokens, config.dim_model
        )
        self.cross_attention = nn.MultiheadAttention(config.dim_model, config.n_heads)
        self.gate = nn.Parameter(torch.tensor(float(config.object_conditioning_gate_init)))
        self.last_debug_shapes: dict[str, tuple[int, ...]] = {}

    def extract_raw_object_tokens(self, top_image: Tensor) -> tuple[Tensor, Tensor]:
        patches, grid_shape = self.dino(top_image)
        raw_tokens, similarity = self.tokenizer(patches, grid_shape, self.prototype_bank.prototypes)
        self.last_debug_shapes = {
            "top_rgb": tuple(top_image.shape),
            "dino_patches": tuple(patches.shape),
            "similarity_map": tuple(similarity.shape),
            "object_tokens": tuple(raw_tokens.shape),
        }
        return raw_tokens, similarity

    def adapt_object_tokens(self, raw_tokens: Tensor) -> Tensor:
        tokens = self.adapter(raw_tokens) + self.token_slot_embedding.weight.unsqueeze(0)
        self.last_debug_shapes["adapter_output"] = tuple(tokens.shape)
        return tokens

    @staticmethod
    def global_spatial_features(raw_tokens: Tensor) -> Tensor:
        """Return the frozen global soft center and confidence as [x, y, c]."""
        if raw_tokens.ndim != 3 or raw_tokens.shape[1] < 1 or raw_tokens.shape[2] < 387:
            raise ValueError(f"Unexpected raw object-token shape: {tuple(raw_tokens.shape)}")
        return raw_tokens[:, 0, 384:387]

    def make_object_tokens(self, top_image: Tensor) -> tuple[Tensor, Tensor]:
        raw_tokens, similarity = self.extract_raw_object_tokens(top_image)
        tokens = self.adapt_object_tokens(raw_tokens)
        return tokens, similarity

    def forward(self, act_tokens: Tensor, top_image: Tensor, object_tokens: Tensor | None = None) -> Tensor:
        if object_tokens is None:
            object_tokens, _ = self.make_object_tokens(top_image)
        key_value = object_tokens.transpose(0, 1)
        response, _ = self.cross_attention(act_tokens, key_value, key_value, need_weights=False)
        self.last_debug_shapes["cross_attention_output"] = tuple(response.shape)
        self.last_debug_shapes["act_encoder_input"] = tuple(act_tokens.shape)
        return act_tokens + self.gate * response
