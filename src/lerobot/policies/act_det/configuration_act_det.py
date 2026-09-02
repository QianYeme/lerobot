#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration for ACTDet — ACT with Detection-guided visual perception."""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_det")
@dataclass
class ACTDetConfig(ACTConfig):
    """Configuration for the ACTDet (Action Chunking Transformer with Detection) policy.

    Extends the standard ACT config with detection branch, mask-guided perception,
    data augmentation, and feature fusion parameters.

    The detection branch adds a Feature Pyramid Network (FPN) and FCOS detection
    head on top of the shared ResNet18 backbone, with detection-guided feature
    fusion to inject spatial attention into the action branch.

    The mask-guided perception branch adds a lightweight Mask Decoder that uses
    SAM 2 pre-generated masks as pixel-wise supervision to teach the backbone
    fine-grained visual features of transparent objects.

    Args:
        use_detection: Master switch for the detection branch.
        det_weight: Weight multiplier for the detection loss in joint training.
        det_cameras: Per-camera detection configuration.
        annotation_dir: Path to CVAT XML annotation directory (None = no labels).
        use_mask_guidance: Master switch for the mask-guided perception branch.
        mask_weight: Weight multiplier for the mask loss.
        mask_dir: Path to SAM 2 pre-generated NPZ masks (None = {annotation_dir}/masks).
        mask_decoder_channels: Hidden channels in the Mask Decoder upsampling path.
        fpn_channels: FPN output channel count.
        fcos_num_classes: Number of object classes (1 = cup only).
        fcos_strides: Strides for each FPN level.
        fcos_size_ranges: Target size ranges for FPN level assignment.
        focal_alpha: Alpha parameter for Focal Loss.
        focal_gamma: Gamma parameter for Focal Loss.
        fusion_hidden: Hidden channels in fusion attention block.
        aug_enable: Master switch for data augmentation.
        aug_probability: Independent application probability per method.
        aug_color_jitter_enable: Enable color jitter augmentation.
        aug_noise_enable: Enable Gaussian noise augmentation.
        aug_occlusion_enable: Enable random occlusion augmentation.
    """

    # --- Detection ---
    use_detection: bool = True
    det_weight: float = 10.0
    det_cameras: dict = field(
        default_factory=lambda: {
            "observation.images.top": {"enable": True},
            "observation.images.wrist": {"enable": False},
        }
    )

    # Annotations directory.
    annotation_dir: str | None = None

    # --- FPN ---
    fpn_channels: int = 128
    fpn_in_channels: list[int] = field(
        default_factory=lambda: [128, 256, 512]
    )

    # --- FCOS ---
    fcos_num_classes: int = 1
    fcos_num_convs: int = 4
    fcos_gn_groups: int = 32
    fcos_strides: list[int] = field(
        default_factory=lambda: [8, 16, 32]
    )
    fcos_size_ranges: list[tuple[int, int]] = field(
        default_factory=lambda: [
            (0, 60),
            (60, 120),
            (120, 99999),
        ]
    )
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # --- Feature Fusion ---
    fusion_hidden: int = 64

    # --- Data Augmentation ---
    aug_enable: bool = True
    aug_probability: float = 0.9
    aug_color_jitter_enable: bool = True
    aug_brightness: tuple[float, float] = (0.8, 1.2)
    aug_contrast: tuple[float, float] = (0.8, 1.2)
    aug_saturation: tuple[float, float] = (0.8, 1.2)
    aug_hue: tuple[float, float] = (-0.1, 0.1)
    aug_noise_enable: bool = True
    aug_noise_std_range: tuple[float, float] = (0.01, 0.05)
    aug_occlusion_enable: bool = True
    aug_occlusion_area_ratio: tuple[float, float] = (0.1, 0.3)
    aug_occlusion_gray_range: tuple[float, float] = (0.3, 0.7)

    # --- Mask-Guided Perception ---
    use_mask_guidance: bool = True
    mask_weight: float = 1.0
    mask_dir: str | None = None
    mask_decoder_channels: int = 32
    mask_cameras: dict = field(
        default_factory=lambda: {
            "observation.images.top": {"enable": True},
            "observation.images.wrist": {"enable": False},
        }
    )

    # --- Feature Injection (Innovation 2 & 3) ---
    # FCOS Feature Injection: extract cls_tower + reg_tower intermediate features,
    # gate with centerness, project to dim_model, and append as extra Encoder tokens.
    fcos_feature_inject: bool = False
    # Which FPN levels to inject FCOS features from. ["p4"] = 300 tokens,
    # ["p3","p4"] = 1500, ["p2","p3","p4"] = 6300.
    fcos_inject_levels: list[str] = field(default_factory=lambda: ["p4"])

    # Mask Feature Injection: extract Mask Decoder f432 intermediate features,
    # project to dim_model, pool, and append as extra Encoder tokens.
    mask_feature_inject: bool = False
    # Spatial resolution to pool the mask inject features to before flattening.
    mask_inject_pool_size: tuple[int, int] = (15, 20)
