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
"""ACTDet: Action Chunking Transformer with Detection-guided visual perception.

Extends the standard ACT model with:
  - Online data augmentation (color jitter, noise, occlusion) for the global camera.
  - A shared Feature Pyramid Network (FPN) on the ResNet18 backbone.
  - An FCOS detection head for object localization.
  - A Mask Decoder for mask-guided perception (transparent object fine features).
  - A detection-guided feature fusion module that injects spatial attention
    into the action branch.

Jointly trained with:
    Total Loss = action_L1 + KL_div + det_weight * detection_loss + mask_weight * mask_loss
"""

from __future__ import annotations

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import (
    ACTDecoder,
    ACTEncoder,
    ACTSinusoidalPositionEmbedding2d,
    create_sinusoidal_pos_embedding,
)
from lerobot.policies.act_det.configuration_act_det import ACTDetConfig
from lerobot.policies.act_det.detection.augmentation import ImageAugmentation
from lerobot.policies.act_det.detection.fcos import (
    FCOSHead,
    compute_fcos_loss,
    decode_fcos_top1_condition,
)
from lerobot.policies.act_det.detection.fpn import FeaturePyramidNetwork
from lerobot.policies.act_det.detection.fusion import DetectionFeatureFusion
from lerobot.policies.act_det.detection.mask_decoder import MaskDecoder, mask_supervision_loss
from lerobot.policies.act_det.label_loader import LabelLoader
from lerobot.policies.act_det.mask_loader import MaskLoader
from lerobot.policies.act_det.water_keypoint import WaterPointLabels, water_keypoint_loss
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class ACTDetPolicy(PreTrainedPolicy):
    """ACTDet Policy: ACT with enhanced visual perception via object detection.

    Configuration class: ACTDetConfig.
    """

    config_class = ACTDetConfig
    name = "act_det"

    def __init__(self, config: ACTDetConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACTDetModel(config)

        if config.temporal_ensemble_coeff is not None:
            # Reuse ACT's temporal ensembler; import deferred to avoid circular refs.
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            from collections import deque
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        l1_loss_per_dim = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none")
            * ~batch["action_is_pad"].unsqueeze(-1)
        )
        if self.config.gripper_loss_weight != 1.0:
            # Up-weight the gripper channel (last action dim) so it doesn't collapse
            # to the mean under the L1 loss (open/close/release are rare events).
            channel_weights = torch.ones(
                actions_hat.shape[-1], device=actions_hat.device, dtype=actions_hat.dtype
            )
            channel_weights[-1] = self.config.gripper_loss_weight
            l1_loss = (l1_loss_per_dim * channel_weights).mean()
        else:
            l1_loss = l1_loss_per_dim.mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp()))
                .sum(-1)
                .mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        # Add detection loss if available.
        det_loss = self.model.get_detection_loss()
        if det_loss is not None and self.config.use_detection:
            det_total, det_components = det_loss
            loss = loss + det_total * self.config.det_weight
            loss_dict.update(det_components)

        # Add mask loss if available.
        mask_loss = self.model.get_mask_loss()
        if mask_loss is not None and getattr(self.config, "use_mask_guidance", False):
            loss = loss + mask_loss["mask_loss"] * getattr(self.config, "mask_weight", 1.0)
            loss_dict.update({key: value.detach().item() for key, value in mask_loss.items()})

        if self.model._water_loss is not None:
            loss = loss + self.model._water_loss["water_keypoint_loss"] * self.config.water_keypoint_weight
            loss_dict.update({key: value.detach().item() for key, value in self.model._water_loss.items()})

        residual, alpha = self.model.get_box_action_residual()
        if residual is not None:
            loss_dict["box_action_residual_alpha"] = alpha.detach().item()
            loss_dict["box_action_residual_abs_mean"] = residual.abs().mean().item()
            loss_dict["box_action_residual_abs_max"] = residual.abs().max().item()

        return loss, loss_dict


class ACTDetModel(nn.Module):
    """The underlying neural network for ACTDetPolicy.

    Extends the standard ACT architecture with:
      - Shared ResNet18 backbone outputting multi-scale features (layer2/3/4).
      - FPN for multi-scale feature fusion.
      - FCOS detection head with Focal Loss, L1 regression, BCE centerness.
      - Detection-guided feature fusion (spatial attention injection).
      - Online data augmentation for the global camera view.
    """

    def __init__(self, config: ACTDetConfig):
        super().__init__()
        self.config = config
        import torchvision
        from torchvision.models._utils import IntermediateLayerGetter
        from torchvision.ops.misc import FrozenBatchNorm2d

        # ---- Shared backbone ----
        if self.config.image_features:
            backbone_model = torchvision.models.__dict__[config.vision_backbone](
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(
                backbone_model,
                return_layers={"layer2": "f2", "layer3": "f3", "layer4": "f4"},
            )
            backbone_out_channels = backbone_model.fc.in_features  # 512 for ResNet18

        # ---- Detection modules ----
        self.use_detection = getattr(config, "use_detection", False)
        if self.use_detection:
            self.fpn = FeaturePyramidNetwork(
                in_channels_list=config.fpn_in_channels,
                out_channels=config.fpn_channels,
            )
            self.fcos_head = FCOSHead(
                in_channels=config.fpn_channels,
                num_classes=config.fcos_num_classes,
                num_convs=config.fcos_num_convs,
                gn_groups=config.fcos_gn_groups,
                inject_dim=config.dim_model if getattr(config, "fcos_feature_inject", False) else None,
            )
            self.fusion = DetectionFeatureFusion(
                fpn_channels=config.fpn_channels,
                f4_in_channels=backbone_out_channels,
                fusion_hidden=config.fusion_hidden,
                out_channels=backbone_out_channels,
            )
            if config.fcos_feature_inject and config.fcos_inject_mode == "residual":
                self.fcos_residual_alpha = nn.Parameter(torch.tensor(config.fcos_residual_alpha))

            # Annotation loader.
            annotation_dir = getattr(config, "annotation_dir", None)
            self.label_loader = LabelLoader(
                annotation_dir=annotation_dir,
                camera_keys={
                    k: k.split(".")[-1] for k in config.det_cameras
                },
            )

            # ---- Mask-Guided Perception ----
            self.use_mask_guidance = getattr(config, "use_mask_guidance", False)
            if self.use_mask_guidance:
                mask_dir = getattr(config, "mask_dir", None) or (
                    f"{annotation_dir}/masks" if annotation_dir else None
                )
                self.mask_loader = MaskLoader(
                    mask_dir=mask_dir,
                    camera_keys={
                        k: k.split(".")[-1] for k in config.det_cameras
                    },
                    max_cache_episodes=getattr(config, "mask_cache_episodes", None),
                    strict=True,
                )
                self.mask_decoder = MaskDecoder(
                    fpn_channels=config.fpn_channels,
                    mid_channels=getattr(config, "mask_decoder_channels", 32),
                    inject_dim=config.dim_model if getattr(config, "mask_feature_inject", False) else None,
                )
            else:
                self.mask_loader = None
                self.mask_decoder = None

        self.water_point_labels = None  # Lazy loading: deployment needs no label file.
        if config.use_water_keypoint:
            if "observation.images.top" not in config.image_features:
                raise ValueError("Water keypoint requires a top image input feature")
            self.water_decoder = MaskDecoder(
                fpn_channels=config.fpn_channels,
                mid_channels=config.mask_decoder_channels,
                output_resolution=(60, 80),
            )

        # ---- Augmentation (top camera only) ----
        self.aug_enable = getattr(config, "aug_enable", False)
        if self.aug_enable:
            self.augmentation = ImageAugmentation(
                probability=config.aug_probability,
                color_jitter_enable=config.aug_color_jitter_enable,
                brightness=config.aug_brightness,
                contrast=config.aug_contrast,
                saturation=config.aug_saturation,
                hue=config.aug_hue,
                gaussian_noise_enable=config.aug_noise_enable,
                noise_std_range=config.aug_noise_std_range,
                random_occlusion_enable=config.aug_occlusion_enable,
                occlusion_area_ratio=config.aug_occlusion_area_ratio,
                occlusion_gray_range=config.aug_occlusion_gray_range,
            )

        # ---- CVAE encoder (same as ACT) ----
        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0], config.dim_model
            )
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # ---- Transformer (same as ACT) ----
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # ---- Encoder input projections ----
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_out_channels, config.dim_model, kernel_size=1
            )

        # Keep shared modules ahead of mode-specific box modules so the same
        # seed produces identical shared initialization across C0/C1/C2.
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])
        self._reset_parameters()

        # ---- Positional embeddings ----
        n_1d_tokens = 1  # latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        if config.use_explicit_box_condition and config.box_condition_mode == "token":
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        if config.use_explicit_box_condition:
            if config.box_condition_mode == "token":
                self.box_condition_input_proj = nn.Sequential(
                    nn.Linear(6, config.dim_model),
                    nn.ReLU(),
                    nn.Linear(config.dim_model, config.dim_model),
                )
            elif config.box_condition_mode == "state":
                self.box_condition_state_proj = nn.Linear(6, config.dim_model, bias=False)
            elif config.box_condition_mode == "action_residual":
                condition_dim = self.config.robot_state_feature.shape[0] + 6
                self.box_condition_action_residual = nn.Sequential(
                    nn.Linear(condition_dim, config.dim_model),
                    nn.ReLU(),
                    nn.Linear(
                        config.dim_model,
                        config.chunk_size * self.config.action_feature.shape[0],
                    ),
                )
                initial_logit = torch.logit(torch.tensor(config.box_action_residual_alpha))
                self.box_condition_action_residual_logit = nn.Parameter(initial_logit)

        # Stash detection and mask loss values computed during forward.
        self._det_loss = None
        self._mask_loss = None
        self._water_loss = None
        self._box_condition = None
        self._box_action_residual = None

    def _reset_parameters(self):
        from itertools import chain
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_detection_loss(self) -> tuple[Tensor, dict[str, float]] | None:
        """Return the detection loss computed during the last forward pass."""
        return self._det_loss

    def get_mask_loss(self) -> dict[str, Tensor] | None:
        """Return the mask loss computed during the last forward pass."""
        return self._mask_loss

    def get_box_condition(self) -> Tensor | None:
        """Return the detached box condition used by the latest forward pass."""
        return self._box_condition

    def get_box_action_residual(self) -> tuple[Tensor | None, Tensor | None]:
        """Return the latest C2 residual action and its effective sigmoid gate."""
        if self.config.box_condition_mode != "action_residual":
            return None, None
        return self._box_action_residual, self.box_condition_action_residual_logit.sigmoid()

    def forward(
        self,
        batch: dict[str, Tensor],
        compute_aux_losses: bool | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """Forward pass through ACTDet.

        Returns:
            actions: (B, chunk_size, action_dim)
            latent_params: (mu, log_sigma_x2) or (None, None)
        """
        training = self.training
        # Offline evaluation needs clean eval-mode action predictions (zero z,
        # no dropout/augmentation) while still measuring detection/mask losses.
        compute_aux_losses = training if compute_aux_losses is None else compute_aux_losses
        self._det_loss = None
        self._mask_loss = None
        self._water_loss = None
        self._box_condition = None
        self._box_action_residual = None

        if self.config.use_vae and training:
            assert ACTION in batch

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]
        device = batch[OBS_STATE].device

        # ---- Latent variable ----
        if self.config.use_vae and ACTION in batch and training:
            cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])
            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)
            pos_embed = self.vae_encoder_pos_enc.clone().detach()
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=device,
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32, device=device)

        # ---- Build Transformer encoder inputs ----
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        base_1d_token_count = 1
        if self.config.robot_state_feature:
            base_1d_token_count += 1
        if self.config.env_state_feature:
            base_1d_token_count += 1
        encoder_in_pos_embed = list(
            self.encoder_1d_feature_pos_embed.weight[:base_1d_token_count].unsqueeze(1)
        )

        robot_state_token_index = None
        if self.config.robot_state_feature:
            robot_state_token_index = len(encoder_in_tokens)
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        # Track all detection losses and targets across cameras.
        all_det_cls_logits = []
        all_det_reg_preds = []
        all_det_ctr_preds = []
        all_det_targets = []
        det_strides = self.config.fcos_strides if self.use_detection else None
        det_size_ranges = self.config.fcos_size_ranges if self.use_detection else None
        image_encoder_in_tokens = []
        image_encoder_in_pos_embed = []
        box_action_residual = None

        # Track mask losses across cameras.
        total_mask_loss = torch.tensor(0.0, device=device)
        total_mask_pixels = 0
        total_mask_available_pixels = 0

        if self.config.image_features:
            image_features = list(self.config.image_features.keys())  # ["observation.images.top", ...]

            for cam_idx, img in enumerate(batch[OBS_IMAGES]):
                cam_key = image_features[cam_idx] if cam_idx < len(image_features) else f"cam_{cam_idx}"
                cam_enabled = (
                    self.use_detection
                    and self.config.det_cameras.get(cam_key, {}).get("enable", False)
                )

                # ---- Optional augmentation (top camera only, training only) ----
                if self.aug_enable and training and cam_idx == 0:
                    # Augment each image in the batch independently.
                    augmented_imgs = []
                    for b in range(img.shape[0]):
                        aug_img = self.augmentation(
                            img[b],
                            normalization_mean=self.config.aug_normalization_mean,
                            normalization_std=self.config.aug_normalization_std,
                        )
                        augmented_imgs.append(aug_img)
                    img = torch.stack(augmented_imgs, dim=0)

                # ---- Backbone ----
                backbone_feats = self.backbone(img)  # {"f2": (B,128,60,80), "f3": (B,256,30,40), "f4": (B,512,15,20)}
                f2, f3, f4 = backbone_feats["f2"], backbone_feats["f3"], backbone_feats["f4"]

                if cam_enabled:
                    # ---- Detection path ----
                    fpn_features = self.fpn([f2, f3, f4])  # [P2, P3, P4]
                    p2, p3, p4 = fpn_features

                    if compute_aux_losses and self.config.use_water_keypoint and cam_key == "observation.images.top":
                        if "episode_index" not in batch or "frame_index" not in batch:
                            raise ValueError("Water keypoint supervision requires episode_index and frame_index")
                        if self.water_point_labels is None:
                            if not self.config.water_keypoint_labels:
                                raise FileNotFoundError("Water keypoint supervision requires a label file")
                            self.water_point_labels = WaterPointLabels(self.config.water_keypoint_labels)
                        labels = self.water_point_labels
                        if tuple(img.shape[-2:]) != (labels.height, labels.width):
                            raise ValueError("Water keypoint labels must match unresized input image dimensions")
                        keys = [(int(ep.item()), int(frame.item()), "top")
                                for ep, frame in zip(batch["episode_index"], batch["frame_index"], strict=True)]
                        target, visible = labels.heatmaps(keys, self.water_decoder.output_resolution, device=device)
                        logits = self.water_decoder(p2, p3, p4, return_logits=True)
                        self._water_loss = {
                            "water_keypoint_loss": water_keypoint_loss(logits, target, visible),
                            "water_visible_frames": visible.sum(),
                            "water_coverage": visible.float().mean(),
                        }

                    # FCOS loss is normally training-only, but can be requested
                    # explicitly by the offline evaluator while the model stays in eval mode.
                    need_fcos_outputs = compute_aux_losses or (
                        self.config.use_explicit_box_condition
                        and cam_key == self.config.box_condition_camera
                    )
                    if need_fcos_outputs:
                        cls_logits, reg_preds, ctr_preds = self.fcos_head(fpn_features)

                    if (
                        self.config.use_explicit_box_condition
                        and cam_key == self.config.box_condition_camera
                    ):
                        box_condition = decode_fcos_top1_condition(
                            cls_logits,
                            reg_preds,
                            ctr_preds,
                            strides=self.config.fcos_strides,
                            image_size=tuple(img.shape[-2:]),
                            score_threshold=self.config.box_condition_score_threshold,
                        ).detach()
                        if training and self.config.box_condition_noise_std > 0:
                            coordinate_noise = torch.randn_like(box_condition[:, :4])
                            coordinate_noise *= self.config.box_condition_noise_std
                            box_condition = box_condition.clone()
                            box_condition[:, :4] += coordinate_noise * box_condition[:, 5:6]
                            box_condition[:, :2].clamp_(-1, 1)
                            box_condition[:, 2:4].clamp_(0, 1)
                        if training and self.config.box_condition_dropout > 0:
                            keep = (
                                torch.rand(
                                    (batch_size, 1), device=box_condition.device
                                ) >= self.config.box_condition_dropout
                            )
                            box_condition = box_condition * keep
                        self._box_condition = box_condition
                        if self.config.box_condition_mode == "token":
                            encoder_in_tokens.append(self.box_condition_input_proj(box_condition))
                            encoder_in_pos_embed.append(
                                self.encoder_1d_feature_pos_embed.weight[base_1d_token_count].unsqueeze(0)
                            )
                        elif self.config.box_condition_mode == "state":
                            if robot_state_token_index is None:
                                raise ValueError("State box conditioning requires a robot state token")
                            encoder_in_tokens[robot_state_token_index] = (
                                encoder_in_tokens[robot_state_token_index]
                                + self.box_condition_state_proj(box_condition)
                            )
                        elif self.config.box_condition_mode == "action_residual":
                            state_box_condition = torch.cat((batch[OBS_STATE], box_condition), dim=-1)
                            box_action_residual = self.box_condition_action_residual(state_box_condition)
                            box_action_residual = box_action_residual.reshape(
                                batch_size,
                                self.config.chunk_size,
                                self.config.action_feature.shape[0],
                            )

                    if compute_aux_losses:
                        all_det_cls_logits.append(cls_logits)
                        all_det_reg_preds.append(reg_preds)
                        all_det_ctr_preds.append(ctr_preds)

                        # Build detection targets from annotations.
                        targets_per_img = self._build_detection_targets(
                            batch, cam_key, img_idx=cam_idx
                        )
                        all_det_targets.append(targets_per_img)

                    # ---- Mask Decoder (per-camera switch) ----
                    mask_cam_enabled = (
                        self.use_mask_guidance
                        and getattr(self.config, "mask_cameras", {}).get(cam_key, {}).get("enable", False)
                    )
                    if compute_aux_losses and mask_cam_enabled:
                        mask_loss_type = self.config.mask_loss_type
                        pred_mask = self.mask_decoder(p2, p3, p4, return_logits=mask_loss_type == "bce_dice")
                        total_mask_available_pixels += pred_mask.numel()

                        # Load SAM 2 GT masks for each image in the batch.
                        gt_masks, valid_masks = self._load_mask_batch(
                            batch, cam_key, img_idx=cam_idx, device=device
                        )
                        valid_pixels = int(valid_masks.sum().item())
                        mask_loss = mask_supervision_loss(pred_mask, gt_masks, mask_loss_type, valid_masks)
                        total_mask_loss = total_mask_loss + mask_loss * max(valid_pixels, 1)
                        total_mask_pixels += valid_pixels

                    # Feature fusion (always run — generates spatial attention during inference too).
                    enhanced_f4 = self.fusion(p2, p3, p4, f4)  # (B, 512, 15, 20)

                    # Project and flatten for Transformer encoder.
                    cam_features = self.encoder_img_feat_input_proj(enhanced_f4)
                    if self.config.fcos_feature_inject and self.config.fcos_inject_mode == "residual":
                        injected = self.fcos_head.get_inject_features(fpn_features, levels=["p4"])[0]
                        if injected.shape != cam_features.shape:
                            raise ValueError("Residual FCOS features must match projected image features")
                        cam_features = cam_features + self.fcos_residual_alpha.clamp(0, 1) * injected
                else:
                    # ---- Standard ACT path (no detection for this camera) ----
                    cam_features = self.encoder_img_feat_input_proj(f4)

                # Positional embedding and flatten.
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                image_encoder_in_tokens.extend(list(cam_features))
                image_encoder_in_pos_embed.extend(list(cam_pos_embed))

                # ---- FCOS Feature Injection (Innovation 2) ----
                # Inject cls+reg tower intermediate features, gated by centerness,
                # as extra Encoder tokens right after this camera's image tokens.
                if (
                    cam_enabled
                    and getattr(self.config, "fcos_feature_inject", False)
                    and self.config.fcos_inject_mode == "tokens"
                ):
                    fcos_inject = self.fcos_head.get_inject_features(
                        fpn_features,
                        levels=getattr(self.config, "fcos_inject_levels", ["p4"]),
                    )
                    for inj_feat in fcos_inject:
                        inj_pos = self.encoder_cam_feat_pos_embed(inj_feat).to(dtype=inj_feat.dtype)
                        inj_flat = einops.rearrange(inj_feat, "b c h w -> (h w) b c")
                        inj_pos_flat = einops.rearrange(inj_pos, "b c h w -> (h w) b c")
                        image_encoder_in_tokens.extend(list(inj_flat))
                        image_encoder_in_pos_embed.extend(list(inj_pos_flat))

                # ---- Mask Feature Injection (Innovation 3) ----
                # Inject Mask Decoder f432 intermediate features (pooled) as
                # extra Encoder tokens right after FCOS inject tokens.
                if (
                    cam_enabled
                    and self.use_mask_guidance
                    and getattr(self.config, "mask_feature_inject", False)
                    and getattr(self.config, "mask_cameras", {}).get(cam_key, {}).get("enable", False)
                ):
                    mask_inject = self.mask_decoder.get_inject_features(
                        p2, p3, p4,
                        pool_size=getattr(self.config, "mask_inject_pool_size", (15, 20)),
                    )
                    mask_pos = self.encoder_cam_feat_pos_embed(mask_inject).to(dtype=mask_inject.dtype)
                    mask_flat = einops.rearrange(mask_inject, "b c h w -> (h w) b c")
                    mask_pos_flat = einops.rearrange(mask_pos, "b c h w -> (h w) b c")
                    image_encoder_in_tokens.extend(list(mask_flat))
                    image_encoder_in_pos_embed.extend(list(mask_pos_flat))

        # ---- Compute detection loss ----
        if (
            compute_aux_losses
            and self.use_detection
            and all_det_cls_logits
            and len(all_det_targets) > 0
            and all_det_targets[0] is not None
        ):
            # Sum over all cameras that had detection enabled.
            total_det_loss = 0.0
            total_det_components = {}
            for cam_cls, cam_reg, cam_ctr, cam_targets in zip(
                all_det_cls_logits, all_det_reg_preds, all_det_ctr_preds, all_det_targets
            ):
                det_loss, det_dict = compute_fcos_loss(
                    cam_cls, cam_reg, cam_ctr, cam_targets,
                    strides=det_strides,
                    size_ranges=det_size_ranges,
                    focal_alpha=self.config.focal_alpha,
                    focal_gamma=self.config.focal_gamma,
                )
                total_det_loss = total_det_loss + det_loss
                for k, v in det_dict.items():
                    total_det_components[k] = total_det_components.get(k, 0.0) + v
            self._det_loss = (total_det_loss, total_det_components)

        # ---- Store mask loss ----
        if total_mask_available_pixels > 0:
            self._mask_loss = {
                "mask_loss": total_mask_loss / max(total_mask_pixels, 1),
                "mask_valid_pixels": torch.tensor(total_mask_pixels, device=device),
                "mask_coverage": torch.tensor(total_mask_pixels / total_mask_available_pixels, device=device),
            }

        # Keep all 1D tokens ahead of spatial tokens even when the condition
        # camera is not the first image feature in dataset order.
        encoder_in_tokens.extend(image_encoder_in_tokens)
        encoder_in_pos_embed.extend(image_encoder_in_pos_embed)

        # ---- Transformer encoder → decoder → action head ----
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        decoder_out = decoder_out.transpose(0, 1)  # (B, S, C)
        actions = self.action_head(decoder_out)
        if box_action_residual is not None:
            alpha = self.box_condition_action_residual_logit.sigmoid()
            actions = actions + alpha * box_action_residual
            self._box_action_residual = box_action_residual.detach()

        return actions, (mu, log_sigma_x2)

    def _build_detection_targets(
        self,
        batch: dict[str, Tensor],
        cam_key: str,
        img_idx: int = 0,
    ) -> list[dict]:
        """Build per-image detection targets by looking up cached annotations.

        Args:
            batch: Training batch containing episode_index and frame_index.
            cam_key: Canonical camera key (e.g. "observation.images.top").
            img_idx: Index of this camera in the image features list.

        Returns:
            List of dicts, one per batch element, with keys "labels" and "bboxes".
            Empty dicts for frames/sequences without annotations.
        """
        # If episode_index/frame_index are not in the batch (e.g. streaming mode),
        # detection targets cannot be built — skip with empty targets.
        if "episode_index" not in batch or "frame_index" not in batch:
            return [{}] * batch[OBS_IMAGES][img_idx].shape[0]

        batch_size = batch[OBS_IMAGES][img_idx].shape[0]
        targets = []

        for b in range(batch_size):
            ep = int(batch["episode_index"][b].item())
            frm = int(batch["frame_index"][b].item())

            labels = self.label_loader.get_labels(cam_key, ep, frm)
            if labels is not None:
                # CVAT XML labels are class-name strings (e.g. "cup"); FCOS expects
                # 0-based integer class indices (0..num_classes-1). Map names to
                # indices (single-class dataset → index 0).
                labels["labels"] = [
                    0 if isinstance(name, str) else int(name)
                    for name in labels.get("labels", [])
                ]
                targets.append(labels)
            else:
                targets.append({})

        return targets

    def _get_img_key(self, cam_idx: int) -> str:
        """Map camera index to its canonical key."""
        features = list(self.config.image_features.keys()) if self.config.image_features else []
        if features and cam_idx < len(features):
            return features[cam_idx]
        return f"cam_{cam_idx}"

    def _load_mask_batch(
        self,
        batch: dict[str, Tensor],
        cam_key: str,
        img_idx: int = 0,
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Load SAM 2 GT masks for each image in the batch.

        Args:
            batch: Training batch containing episode_index and frame_index.
            cam_key: Canonical camera key (e.g. "observation.images.top").
            img_idx: Camera index.
            device: Target device for the output tensor.

        Returns:
            GT masks and per-pixel validity, intersected with frame validity.
        """
        if self.mask_loader is None or not self.mask_loader.enabled:
            raise FileNotFoundError("MASK supervision requested without an available mask directory")

        # Auxiliary supervision needs original dataset indices after preprocessing.
        if "episode_index" not in batch or "frame_index" not in batch:
            raise ValueError("MASK supervision requires episode_index and frame_index")

        batch_size = batch[OBS_IMAGES][img_idx].shape[0]
        masks_per_image = []
        valid_per_image = []
        mask_shape = tuple(self.mask_decoder.output_resolution)

        for b in range(batch_size):
            ep = int(batch["episode_index"][b].item())
            frm = int(batch["frame_index"][b].item())

            mask, valid_pixels = self.mask_loader.get_mask_and_valid_pixels(cam_key, ep, frm)
            if mask is None:
                masks_per_image.append(torch.zeros((1, *mask_shape)))
                valid_per_image.append(torch.zeros((1, *mask_shape), dtype=torch.bool))
                continue  # Only explicitly invalid frames may return None in strict mode.
            if mask.shape != mask_shape:
                raise ValueError(f"MASK shape {mask.shape} != decoder shape {mask_shape}: {cam_key}, {ep}, {frm}")

            # mask is (H, W) numpy float32 → (1, H, W) torch.
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
            masks_per_image.append(mask_tensor)
            valid_per_image.append(torch.from_numpy(valid_pixels).unsqueeze(0))

        gt_masks = torch.stack(masks_per_image, dim=0)  # (B, 1, 480, 640)

        if device is not None:
            gt_masks = gt_masks.to(device, non_blocking=True)

        return gt_masks, torch.stack(valid_per_image, dim=0).to(device=device)
