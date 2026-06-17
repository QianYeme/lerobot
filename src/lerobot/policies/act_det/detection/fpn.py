"""Feature Pyramid Network for multi-scale feature fusion.

As described in:
  Lin et al. "Feature Pyramid Networks for Object Detection" (CVPR 2017)

Takes ResNet18's layer2 (F2), layer3 (F3), layer4 (F4) as input and produces
P2, P3, P4 with unified channel dimension.
"""

import torch
from torch import nn


class FeaturePyramidNetwork(nn.Module):
    """FPN that fuses multi-scale backbone features into a semantic pyramid.

    Input:
        f2: (B, 128, 60, 80)  — ResNet18 layer2 output
        f3: (B, 256, 30, 40)  — ResNet18 layer3 output
        f4: (B, 512, 15, 20)  — ResNet18 layer4 output

    Output:
        p2: (B, out_channels, 60, 80)
        p3: (B, out_channels, 30, 40)
        p4: (B, out_channels, 15, 20)
    """

    def __init__(self, in_channels_list: list[int] = (128, 256, 512), out_channels: int = 128):
        """Args:
            in_channels_list: Channel counts of input features [C2, C3, C4].
            out_channels: Unified channel count for output pyramid features.
        """
        super().__init__()
        self.out_channels = out_channels

        # Lateral connections: 1x1 conv to unify channels.
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_c, out_channels, kernel_size=1)
            for in_c in in_channels_list
        ])

        # Smoothing: 3x3 conv after fusion to reduce aliasing from upsampling.
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        for modules in [self.lateral_convs, self.smooth_convs]:
            for m in modules:
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Build feature pyramid via top-down pathway with lateral connections.

        Args:
            features: List of [f2, f3, f4] feature maps from backbone.

        Returns:
            List of [p2, p3, p4] fused feature maps.
        """
        assert len(features) == len(self.lateral_convs), (
            f"Expected {len(self.lateral_convs)} input features, got {len(features)}"
        )

        # Apply lateral (1x1) convolutions.
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway: start from highest-level feature and fuse downwards.
        num_levels = len(laterals)
        for i in range(num_levels - 1, 0, -1):
            # Upsample higher-level feature to match lower-level spatial size.
            upsampled = nn.functional.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        # Apply 3x3 smoothing convolution to each level.
        out = [smooth(lat) for smooth, lat in zip(self.smooth_convs, laterals)]

        return out  # [P2, P3, P4]
