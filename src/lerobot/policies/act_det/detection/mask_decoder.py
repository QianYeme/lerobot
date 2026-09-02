"""Mask Decoder for mask-guided visual perception.

A lightweight segmentation decoder that predicts a pixel-wise mask from
FPN pyramid features. Only runs during training — its purpose is to
supervise the shared backbone with dense pixel-level signals, forcing
the ResNet18 to learn fine visual features of transparent objects.

After training, the Mask Decoder is discarded; only the backbone and
FPN weights (now "sculpted" by mask supervision) are used for inference.

Architecture:
    P4(128ch) → 1×1→32ch → upsample×2 ─┐
                                         ├→ concat → 3×3→32ch
    P3(128ch) → 1×1→32ch → upsample×2 ─┘ │
                                          ├→ upsample×2 ─┐
                                          │               ├→ concat → 3×3→32ch
    P2(128ch) → 1×1→32ch ─────────────────┘               │
                                                           ├→ ×8 upsample
                                                           └→ 1×1→1ch → Sigmoid
                                                              (480×640)
"""

from torch import nn
import torch


class MaskDecoder(nn.Module):
    """Lightweight FPN-based segmentation decoder.

    Takes FPN's P2, P3, P4 feature maps and predicts a single-channel
    probability mask at the original image resolution.

    Args:
        fpn_channels: Number of channels in FPN outputs (default 128).
        mid_channels: Number of channels in the upsampling path (default 32).
        output_resolution: Target (H, W) of the predicted mask.
        inject_dim: If set, adds a 1×1 Conv projection (mid_channels → inject_dim)
            for mask feature injection (Innovation 3). Default None = no injection.
    """

    def __init__(
        self,
        fpn_channels: int = 128,
        mid_channels: int = 32,
        output_resolution: tuple[int, int] = (480, 640),
        inject_dim: int | None = None,
    ):
        super().__init__()
        self.output_resolution = output_resolution
        self.inject_dim = inject_dim

        # Channel reduction: 128 → mid_channels for each FPN level.
        self.reduce_p4 = nn.Conv2d(fpn_channels, mid_channels, kernel_size=1)
        self.reduce_p3 = nn.Conv2d(fpn_channels, mid_channels, kernel_size=1)
        self.reduce_p2 = nn.Conv2d(fpn_channels, mid_channels, kernel_size=1)

        # Fusion convs after concatenation.
        # P4_up + P3_reduced → 2×mid_channels → mid_channels.
        self.fuse_43 = nn.Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # (P4+P3)_up + P2_reduced → 2×mid_channels → mid_channels.
        self.fuse_432 = nn.Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Final prediction: mid_channels → 1.
        self.predict = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Injection projection: mid_channels → inject_dim (Innovation 3).
        if inject_dim is not None:
            self.inject_proj = nn.Conv2d(mid_channels, inject_dim, kernel_size=1)

        self._reset_parameters()

    def _reset_parameters(self):
        """Kaiming init for convolution layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Inject projection gets lighter init to avoid disrupting the encoder.
        if self.inject_dim is not None:
            nn.init.normal_(self.inject_proj.weight, std=0.01)
            nn.init.constant_(self.inject_proj.bias, 0)

    def forward(self, p2, p3, p4):
        """Predict pixel-wise mask from FPN features.

        Args:
            p2: (B, C, 60, 80)  — FPN P2 feature.
            p3: (B, C, 30, 40)  — FPN P3 feature.
            p4: (B, C, 15, 20)  — FPN P4 feature.

        Returns:
            mask: (B, 1, H_out, W_out) predicted mask, values in [0, 1].
        """
        # Reduce channels.
        r4 = self.reduce_p4(p4)  # (B, 32, 15, 20)
        r3 = self.reduce_p3(p3)  # (B, 32, 30, 40)
        r2 = self.reduce_p2(p2)  # (B, 32, 60, 80)

        # P4 → upsample → concat with P3 → fuse.
        u4 = nn.functional.interpolate(r4, size=r3.shape[-2:], mode="bilinear", align_corners=False)
        f43 = self.fuse_43(torch.cat([u4, r3], dim=1))  # (B, 32, 30, 40)

        # → upsample → concat with P2 → fuse.
        u43 = nn.functional.interpolate(f43, size=r2.shape[-2:], mode="bilinear", align_corners=False)
        f432 = self.fuse_432(torch.cat([u43, r2], dim=1))  # (B, 32, 60, 80)

        # → upsample ×8 to target resolution.
        h, w = self.output_resolution
        up = nn.functional.interpolate(f432, size=(h, w), mode="bilinear", align_corners=False)

        # → final prediction.
        return self.predict(up)  # (B, 1, 480, 640)

    def get_inject_features(
        self,
        p2: "torch.Tensor",
        p3: "torch.Tensor",
        p4: "torch.Tensor",
        pool_size: tuple[int, int] = (15, 20),
    ) -> "torch.Tensor":
        """Extract the fused intermediate feature f432 for injection (Innovation 3).

        Computes the 32-dim compressed semantic f432 (P2/P3/P4 fully fused,
        before ×8 upsampling and prediction), then:
          1. Projects 32 → inject_dim via 1×1 Conv.
          2. adaptive_avg_pool2d to ``pool_size`` to control token count.

        Args:
            p2: FPN P2 feature (B, C, 60, 80).
            p3: FPN P3 feature (B, C, 30, 40).
            p4: FPN P4 feature (B, C, 15, 20).
            pool_size: Target (H, W) for adaptive pooling. Default (15, 20) = 300 tokens.

        Returns:
            Projected and pooled tensor (B, inject_dim, pool_H, pool_W).
        """
        import torch.nn.functional as F  # noqa: N812

        if self.inject_dim is None:
            raise ValueError(
                "MaskDecoder was not initialized with inject_dim. "
                "Set inject_dim in the constructor to use feature injection."
            )

        # Reduce channels (shared with forward).
        r4 = self.reduce_p4(p4)  # (B, 32, 15, 20)
        r3 = self.reduce_p3(p3)  # (B, 32, 30, 40)
        r2 = self.reduce_p2(p2)  # (B, 32, 60, 80)

        # Fuse P4 + P3.
        u4 = F.interpolate(r4, size=r3.shape[-2:], mode="bilinear", align_corners=False)
        f43 = self.fuse_43(torch.cat([u4, r3], dim=1))  # (B, 32, 30, 40)

        # Fuse (P4+P3) + P2 → f432.
        u43 = F.interpolate(f43, size=r2.shape[-2:], mode="bilinear", align_corners=False)
        f432 = self.fuse_432(torch.cat([u43, r2], dim=1))  # (B, 32, 60, 80)

        # Project to inject_dim.
        projected = self.inject_proj(f432)  # (B, inject_dim, 60, 80)

        # Pool to control token count.
        pooled = F.adaptive_avg_pool2d(projected, pool_size)  # (B, inject_dim, H_pool, W_pool)

        return pooled
