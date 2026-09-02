"""FCOS detection head and loss functions.

Implements the FCOS (Fully Convolutional One-Stage Object Detection) head as
described in:
  Tian et al. "FCOS: Fully Convolutional One-Stage Object Detection" (ICCV 2019)

The detection head consists of three parallel convolutional towers:
  - Classification tower: predicts per-pixel class logits.
  - Regression tower: predicts distances to bbox edges (l, t, r, b).
  - Centerness tower: predicts centerness score to suppress low-quality detections.

All three towers share the same architecture across FPN levels with shared weights.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class FCOSHead(nn.Module):
    """FCOS detection head with shared weights across FPN levels.

    Each tower consists of 4 layers of 3x3 conv + GroupNorm + ReLU,
    followed by a final prediction layer.

    Args:
        in_channels: Number of input channels from FPN.
        num_classes: Number of object classes (excluding background).
        num_convs: Number of convolutional layers in each tower.
        gn_groups: Number of groups for GroupNorm.
        inject_dim: If set, adds a 1×1 Conv projection (256 → inject_dim) for
            feature injection (Innovation 2). Default None = no injection.
    """

    def __init__(
        self,
        in_channels: int = 128,
        num_classes: int = 1,
        num_convs: int = 4,
        gn_groups: int = 32,
        inject_dim: int | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_convs = num_convs
        self.inject_dim = inject_dim

        # Shared feature towers (one per branch).
        self.cls_feature = self._make_tower(in_channels)
        self.reg_feature = self._make_tower(in_channels)
        self.ctr_feature = self._make_tower(in_channels)

        # Prediction layers.
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.ctr_pred = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        # Scale parameters for regression (one per FPN level, learned).
        self.scales = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(3)
        ])

        # Injection projection: 256 → inject_dim (Innovation 2).
        if inject_dim is not None:
            self.inject_proj = nn.Conv2d(in_channels * 2, inject_dim, kernel_size=1)

        self._reset_parameters()

    def _make_tower(self, in_channels: int) -> nn.Sequential:
        """Build a tower of num_convs layers with GroupNorm and ReLU."""
        layers = []
        for _ in range(self.num_convs):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            layers.append(nn.GroupNorm(min(32, in_channels), in_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _reset_parameters(self):
        """Initialize with normal distribution (std=0.01) as per paper."""
        for modules in [self.cls_feature, self.reg_feature, self.ctr_feature]:
            for m in modules:
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.constant_(m.bias, 0)

        # Classification bias: -log((1 - prior) / prior) for prior=0.01.
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1.0 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_logits.bias, bias_value)
        nn.init.normal_(self.cls_logits.weight, std=0.01)

        nn.init.normal_(self.reg_pred.weight, std=0.01)
        nn.init.constant_(self.reg_pred.bias, 0)

        nn.init.normal_(self.ctr_pred.weight, std=0.01)
        nn.init.constant_(self.ctr_pred.bias, 0)

        # Injection projection (Innovation 2).
        if self.inject_dim is not None:
            nn.init.normal_(self.inject_proj.weight, std=0.01)
            nn.init.constant_(self.inject_proj.bias, 0)

    def forward(self, features: list[Tensor]) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """Forward pass over multiple FPN feature levels.

        Args:
            features: List of [P2, P3, P4] feature maps, each (B, C, H, W).

        Returns:
            cls_logits: List of (B, num_classes, H, W) classification logits per level.
            reg_preds:  List of (B, 4, H, W) regression predictions per level.
            ctr_preds:  List of (B, 1, H, W) centerness predictions per level.
        """
        cls_logits_list = []
        reg_preds_list = []
        ctr_preds_list = []

        for i, feat in enumerate(features):
            cls_feat = self.cls_feature(feat)
            reg_feat = self.reg_feature(feat)
            ctr_feat = self.ctr_feature(feat)

            cls_logits_list.append(self.cls_logits(cls_feat))
            # Regression with scale (exp(s_i) * prediction).
            reg_preds_list.append(
                torch.exp(self.scales[i]) * self.reg_pred(reg_feat).exp()
            )
            ctr_preds_list.append(self.ctr_pred(ctr_feat))

        return cls_logits_list, reg_preds_list, ctr_preds_list

    # Mapping from level name to index in the feature list [P2, P3, P4].
    _LEVEL_TO_IDX: dict[str, int] = {"p2": 0, "p3": 1, "p4": 2}

    def get_inject_features(
        self,
        features: list[Tensor],
        levels: list[str],
    ) -> list[Tensor]:
        """Extract FCOS tower intermediate features for injection (Innovation 2).

        For each requested FPN level:
          1. Runs cls_tower, reg_tower, ctr_tower on the feature map.
          2. Concatenates cls_f (128ch) + reg_f (128ch) → 256ch.
          3. Gates with centerness: ×(1.0 + sigmoid(ctr_pred)), so
             high-confidence regions are amplified up to 2×.
          4. Projects 256 → inject_dim via 1×1 Conv.

        Args:
            features: List of [P2, P3, P4] FPN feature maps, each (B, C, H, W).
            levels: Which levels to extract, e.g. ["p4"] or ["p3", "p4"].

        Returns:
            List of projected feature tensors (B, inject_dim, H, W), one per
            requested level, in the same order as ``levels``.
        """
        if self.inject_dim is None:
            raise ValueError(
                "FCOSHead was not initialized with inject_dim. "
                "Set inject_dim in the constructor to use feature injection."
            )

        level_indices = [self._LEVEL_TO_IDX[l] for l in levels]
        inject_tokens = []

        for idx in level_indices:
            feat = features[idx]
            cls_feat = self.cls_feature(feat)  # (B, 128, H, W)
            reg_feat = self.reg_feature(feat)  # (B, 128, H, W)
            ctr_feat = self.ctr_feature(feat)  # (B, 128, H, W)

            # Centerness gate (raw logit → sigmoid → [0,1], then +1 → [1,2]).
            ctr_pred = self.ctr_pred(ctr_feat)  # (B, 1, H, W)
            gate = 1.0 + torch.sigmoid(ctr_pred)  # values in [1, 2]

            # Concatenate cls + reg intermediate features.
            combined = torch.cat([cls_feat, reg_feat], dim=1)  # (B, 256, H, W)
            gated = combined * gate

            # Project to inject_dim.
            projected = self.inject_proj(gated)  # (B, inject_dim, H, W)
            inject_tokens.append(projected)

        return inject_tokens


def compute_fcos_loss(
    cls_logits: list[Tensor],
    reg_preds: list[Tensor],
    ctr_preds: list[Tensor],
    targets: list[dict],
    strides: list[int],
    size_ranges: list[tuple[int, int]],
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> tuple[Tensor, dict[str, float]]:
    """Compute FCOS detection loss across all FPN levels.

    Args:
        cls_logits: Per-level classification logits [(B, C, H, W), ...].
        reg_preds:  Per-level regression predictions [(B, 4, H, W), ...].
        ctr_preds:  Per-level centerness predictions [(B, 1, H, W), ...].
        targets:    List of per-image target dicts.
                    Each dict has keys "labels" (list[int]), "bboxes" (list[[x1,y1,x2,y2]]).
                    Padding images have empty dicts: {}.
        strides:    Feature stride for each FPN level (e.g., [8, 16, 32]).
        size_ranges: Target size range for each level [(lo, hi), ...].
        focal_alpha: Alpha parameter for Focal Loss.
        focal_gamma: Gamma parameter for Focal Loss.

    Returns:
        total_loss: Scalar detection loss.
        loss_dict:  Dict of individual loss components for logging.
    """
    device = cls_logits[0].device
    num_levels = len(cls_logits)
    batch_size = cls_logits[0].shape[0]
    num_classes = cls_logits[0].shape[1]

    # Gather per-level outputs into flat lists of (N, C) or (N, 4) or (N, 1).
    all_cls_logits = []
    all_reg_preds = []
    all_ctr_preds = []
    all_cls_targets = []
    all_reg_targets = []
    all_ctr_targets = []

    for level_idx in range(num_levels):
        stride = strides[level_idx]
        lo, hi = size_ranges[level_idx]
        cls_logit = cls_logits[level_idx]  # (B, C, H, W)
        reg_pred = reg_preds[level_idx]    # (B, 4, H, W)
        ctr_pred = ctr_preds[level_idx]    # (B, 1, H, W)

        _, _, H, W = cls_logit.shape

        # Build per-pixel (x, y) coordinate grid.
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        # Center coordinates of each pixel in the original image space.
        coords_x = (xs + 0.5) * stride  # (H, W)
        coords_y = (ys + 0.5) * stride  # (H, W)

        for img_idx in range(batch_size):
            target = targets[img_idx] if img_idx < len(targets) else {}

            # Per-pixel targets for this image.
            cls_target = torch.zeros((H, W), dtype=torch.long, device=device)
            reg_target = torch.zeros((H, W, 4), dtype=torch.float32, device=device)
            ctr_target = torch.zeros((H, W), dtype=torch.float32, device=device)

            if target and "bboxes" in target and len(target["bboxes"]) > 0:
                bboxes = target["bboxes"]  # list of [x1, y1, x2, y2]
                labels = target.get("labels", [0] * len(bboxes))

                for bbox, label in zip(bboxes, labels):
                    x1, y1, x2, y2 = bbox
                    w, h = x2 - x1, y2 - y1
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                    # Determine which pixels are "in" the bbox center region.
                    # Each pixel (at center coords_x[i,j], coords_y[i,j]) is a positive
                    # sample if it falls within the bbox.
                    in_bbox = (
                        (coords_x >= x1) & (coords_x <= x2) &
                        (coords_y >= y1) & (coords_y <= y2)
                    )

                    # Scale-range check: only assign to this FPN level if the bbox
                    # size falls within [lo, hi).
                    bbox_max_dim = max(w, h)
                    if bbox_max_dim < lo or bbox_max_dim >= hi:
                        in_bbox = torch.zeros_like(in_bbox)

                    if not in_bbox.any():
                        continue

                    # Distances to four edges for positive pixels.
                    l_target = coords_x[in_bbox] - x1
                    t_target = coords_y[in_bbox] - y1
                    r_target = x2 - coords_x[in_bbox]
                    b_target = y2 - coords_y[in_bbox]

                    cls_target[in_bbox] = label + 1  # 0 = background, 1..C = classes
                    reg_target[in_bbox, 0] = l_target
                    reg_target[in_bbox, 1] = t_target
                    reg_target[in_bbox, 2] = r_target
                    reg_target[in_bbox, 3] = b_target

                    # Centerness = sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b)).
                    lr_min = torch.min(l_target, r_target)
                    lr_max = torch.max(l_target, r_target)
                    tb_min = torch.min(t_target, b_target)
                    tb_max = torch.max(t_target, b_target)
                    ctr_target[in_bbox] = torch.sqrt(
                        (lr_min / (lr_max + 1e-8)) * (tb_min / (tb_max + 1e-8))
                    )

            # Flatten spatial dims and collect.
            all_cls_logits.append(cls_logit[img_idx].permute(1, 2, 0).reshape(-1, num_classes))
            all_reg_preds.append(reg_pred[img_idx].permute(1, 2, 0).reshape(-1, 4))
            all_ctr_preds.append(ctr_pred[img_idx].reshape(-1))
            all_cls_targets.append(cls_target.reshape(-1))
            all_reg_targets.append(reg_target.reshape(-1, 4))
            all_ctr_targets.append(ctr_target.reshape(-1))

    cls_logits_flat = torch.cat(all_cls_logits, dim=0)  # (N_total, C)
    reg_preds_flat = torch.cat(all_reg_preds, dim=0)     # (N_total, 4)
    ctr_preds_flat = torch.cat(all_ctr_preds, dim=0)     # (N_total,)
    cls_targets_flat = torch.cat(all_cls_targets, dim=0) # (N_total,)
    reg_targets_flat = torch.cat(all_reg_targets, dim=0) # (N_total, 4)
    ctr_targets_flat = torch.cat(all_ctr_targets, dim=0) # (N_total,)

    # --- Classification loss (Focal Loss) ---
    pos_mask = cls_targets_flat > 0
    num_pos = max(pos_mask.sum().item(), 1)

    # One-hot encode targets for Focal Loss.
    cls_targets_onehot = torch.zeros(
        cls_targets_flat.shape[0], num_classes + 1, device=device
    )
    cls_targets_onehot.scatter_(1, cls_targets_flat.unsqueeze(1), 1)
    cls_targets_onehot = cls_targets_onehot[:, 1:]  # Remove background class.

    # Convert logits to probabilities.
    cls_probs = cls_logits_flat.sigmoid()

    # Focal Loss: -α_t * (1 - p_t)^γ * log(p_t).
    alpha_factor = torch.where(
        cls_targets_onehot == 1,
        torch.tensor(focal_alpha, device=device),
        torch.tensor(1 - focal_alpha, device=device),
    )
    focal_weight = torch.where(
        cls_targets_onehot == 1,
        (1 - cls_probs).pow(focal_gamma),
        cls_probs.pow(focal_gamma),
    )
    bce = F.binary_cross_entropy_with_logits(
        cls_logits_flat, cls_targets_onehot, reduction="none"
    )
    cls_loss = (alpha_factor * focal_weight * bce).sum() / num_pos

    # --- Regression loss (L1, positive samples only) ---
    if pos_mask.any():
        reg_loss = F.l1_loss(
            reg_preds_flat[pos_mask], reg_targets_flat[pos_mask], reduction="mean"
        )
    else:
        reg_loss = torch.tensor(0.0, device=device)

    # --- Centerness loss (BCE, positive samples only) ---
    if pos_mask.any():
        ctr_loss = F.binary_cross_entropy_with_logits(
            ctr_preds_flat[pos_mask], ctr_targets_flat[pos_mask], reduction="mean"
        )
    else:
        ctr_loss = torch.tensor(0.0, device=device)

    total_loss = cls_loss + reg_loss + ctr_loss

    loss_dict = {
        "det_cls_loss": cls_loss.item(),
        "det_reg_loss": reg_loss.item(),
        "det_ctr_loss": ctr_loss.item(),
        "det_num_pos": float(num_pos),
    }

    return total_loss, loss_dict
