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
        mask_cache_episodes: LRU cap for cached mask arrays (None = cache all, ~40 GB RAM for formal1_B).
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
        aug_normalization_mean: Image normalization mean used by the saved preprocessor.
        aug_normalization_std: Image normalization std used by the saved preprocessor.
    """

    # --- Detection ---
    use_detection: bool = True
    # Detection loss weight in the joint loss. Was 10.0 under the (buggy) unnormalized
    # pixel regression scale, which let det_total (~30) dominate the action L1 (~0.7)
    # by ~400x. With the regression targets now normalized by stride, det_total drops to
    # single digits, so weight 1.0 keeps detection as a meaningful but not dominant
    # auxiliary signal (FCOS paper uses balance lambda=1). Tune 0.1-1.0 if needed.
    det_weight: float = 1.0
    # Note: `gripper_loss_weight` (action L1 loss weighting) is inherited from ACTConfig.
    det_cameras: dict = field(
        default_factory=lambda: {
            "observation.images.top": {"enable": True},
            "observation.images.gripper": {"enable": False},
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
    # The model receives normalized images. Augmentation temporarily maps them
    # back to RGB [0, 1], applies pixel-space transforms, then normalizes again.
    # These defaults match training with `use_imagenet_stats=True`.
    aug_normalization_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    aug_normalization_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # --- Mask-Guided Perception ---
    use_mask_guidance: bool = True
    mask_weight: float = 1.0
    mask_loss_type: str = "l1"
    mask_dir: str | None = None
    mask_cache_episodes: int | None = None
    mask_decoder_channels: int = 32
    mask_cameras: dict = field(
        default_factory=lambda: {
            "observation.images.top": {"enable": True},
            "observation.images.gripper": {"enable": False},
        }
    )

    # --- Feature Injection (Innovation 2 & 3) ---
    # FCOS Feature Injection: extract cls_tower + reg_tower intermediate features,
    # gate with centerness, project to dim_model, and append as extra Encoder tokens.
    fcos_feature_inject: bool = False
    fcos_inject_mode: str = "tokens"
    fcos_residual_alpha: float = 0.05
    # Which FPN levels to inject FCOS features from. ["p4"] = 300 tokens,
    # ["p3","p4"] = 1500, ["p2","p3","p4"] = 6300.
    fcos_inject_levels: list[str] = field(default_factory=lambda: ["p4"])

    # Explicit FCOS box conditioning (C0): decode the top-scoring predicted box,
    # stop action gradients at the decoded values, and encode the normalized
    # [cx, cy, w, h, confidence, visible] vector as one Transformer token.
    use_explicit_box_condition: bool = False
    box_condition_mode: str = "token"
    box_condition_camera: str = "observation.images.top"
    box_condition_score_threshold: float = 0.25
    box_condition_dropout: float = 0.0
    box_condition_noise_std: float = 0.0
    box_action_residual_alpha: float = 0.05

    # Mask Feature Injection: extract Mask Decoder f432 intermediate features,
    # project to dim_model, pool, and append as extra Encoder tokens.
    mask_feature_inject: bool = False
    # Spatial resolution to pool the mask inject features to before flattening.
    mask_inject_pool_size: tuple[int, int] = (15, 20)

    # Optional visible water-reference heatmap auxiliary task, top camera only.
    use_water_keypoint: bool = False
    water_keypoint_labels: str | None = None
    water_keypoint_weight: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        if self.use_water_keypoint:
            if not self.use_detection or self.use_mask_guidance or self.mask_feature_inject:
                raise ValueError("Water keypoint requires DET and disables MASK supervision/injection")
            if not self.det_cameras.get("observation.images.top", {}).get("enable", False):
                raise ValueError("Water keypoint requires enabled top detection camera")
            if self.aug_enable and self.aug_occlusion_enable:
                raise ValueError("Water keypoint pilot requires random occlusion augmentation disabled")
            if self.water_keypoint_weight < 0:
                raise ValueError("water_keypoint_weight must be nonnegative")
        if self.mask_loss_type not in ("l1", "bce_dice"):
            raise ValueError("mask_loss_type must be 'l1' or 'bce_dice'")
        if self.fcos_inject_mode not in ("tokens", "residual"):
            raise ValueError("fcos_inject_mode must be 'tokens' or 'residual'")
        if not 0 <= self.fcos_residual_alpha <= 1:
            raise ValueError("fcos_residual_alpha must be in [0, 1]")
        if self.fcos_inject_mode == "residual" and self.fcos_inject_levels != ["p4"]:
            raise ValueError("Residual injection requires exactly the p4 level")
        if self.use_explicit_box_condition:
            if not self.use_detection:
                raise ValueError("Explicit box conditioning requires detection")
        if self.box_condition_mode not in ("token", "state", "action_residual"):
            raise ValueError("box_condition_mode must be 'token', 'state', or 'action_residual'")
        if not 0 <= self.box_condition_score_threshold <= 1:
            raise ValueError("box_condition_score_threshold must be in [0, 1]")
        if not 0 <= self.box_condition_dropout <= 1:
            raise ValueError("box_condition_dropout must be in [0, 1]")
        if self.box_condition_noise_std < 0:
            raise ValueError("box_condition_noise_std must be nonnegative")
        if not 0 < self.box_action_residual_alpha < 1:
            raise ValueError("box_action_residual_alpha must be strictly between 0 and 1")

    def validate_features(self) -> None:
        super().validate_features()
        if self.use_explicit_box_condition:
            if self.box_condition_camera not in self.image_features:
                raise ValueError("Box condition camera must be an image input feature")
            if not self.det_cameras.get(self.box_condition_camera, {}).get("enable", False):
                raise ValueError("Box condition camera must have detection enabled")
            if self.box_condition_mode in ("state", "action_residual") and self.robot_state_feature is None:
                raise ValueError(f"Box condition mode '{self.box_condition_mode}' requires robot state")
