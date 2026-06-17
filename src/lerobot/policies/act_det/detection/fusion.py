"""Detection-guided feature fusion module.

Generates a spatial attention map from FPN pyramid features and uses it to enhance
the ResNet backbone's highest-level feature (F4) before it enters the Transformer encoder.
"""

import torch
from torch import nn


class DetectionFeatureFusion(nn.Module):
    """Fuses multi-scale FPN features into a spatial attention map that enhances F4.

    Architecture (from paper Section 3.3.3):
        1. Upsample P2(x4) and P3(x2) to match P4 resolution (15x20).
        2. Concatenate along channels → (B, 3C, 15, 20).
        3. Fusion conv → 64 channels → ReLU → 1x1 conv → 1 channel → Sigmoid.
           Produces spatial attention map A ∈ (B, 1, 15, 20).
        4. Upsample both A(x4) and F4(x4) to 60x80, concatenate.
        5. Enhancement network (2x 3x3 conv + BN + ReLU) → 512 channels.
        6. Downsample back to 15x20, ready for Transformer encoder.
    """

    def __init__(
        self,
        fpn_channels: int = 128,
        f4_in_channels: int = 512,
        fusion_hidden: int = 64,
        out_channels: int = 512,
    ):
        """Args:
            fpn_channels: Channel count of each FPN output (P2/P3/P4).
            f4_in_channels: Channel count of ResNet18 F4 (layer4 output).
            fusion_hidden: Hidden channels in the attention-generation conv block.
            out_channels: Output channels of the enhanced feature map.
        """
        super().__init__()
        self.fpn_channels = fpn_channels
        self.out_channels = out_channels

        # Attention generation: 3C → fusion_hidden → 1
        self.attn_conv1 = nn.Conv2d(fpn_channels * 3, fusion_hidden, kernel_size=3, padding=1)
        self.attn_relu = nn.ReLU(inplace=True)
        self.attn_conv2 = nn.Conv2d(fusion_hidden, 1, kernel_size=1)
        self.attn_sigmoid = nn.Sigmoid()

        # Enhancement network: F4(512) + attention(1) → 512
        self.enhance_conv1 = nn.Conv2d(f4_in_channels + 1, out_channels, kernel_size=3, padding=1)
        self.enhance_bn1 = nn.BatchNorm2d(out_channels)
        self.enhance_relu1 = nn.ReLU(inplace=True)
        self.enhance_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.enhance_bn2 = nn.BatchNorm2d(out_channels)
        self.enhance_relu2 = nn.ReLU(inplace=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in [self.attn_conv1, self.attn_conv2, self.enhance_conv1, self.enhance_conv2]:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        for m in [self.enhance_bn1, self.enhance_bn2]:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(
        self,
        p2: torch.Tensor,
        p3: torch.Tensor,
        p4: torch.Tensor,
        f4: torch.Tensor,
    ) -> torch.Tensor:
        """Generate spatial attention and enhance F4.

        Args:
            p2: (B, C, 60, 80) FPN P2 feature.
            p3: (B, C, 30, 40) FPN P3 feature.
            p4: (B, C, 15, 20) FPN P4 feature.
            f4: (B, 512, 15, 20) ResNet18 layer4 raw feature.

        Returns:
            enhanced_f4: (B, out_channels, 15, 20) Enhanced feature map
                         ready for flattening into Transformer tokens.
        """
        target_h, target_w = p4.shape[-2:]  # 15, 20

        # Step 1: Upsample P2 and P3 to match P4 resolution.
        p2_up = nn.functional.interpolate(p2, size=(target_h, target_w), mode="bilinear", align_corners=False)
        p3_up = nn.functional.interpolate(p3, size=(target_h, target_w), mode="bilinear", align_corners=False)

        # Step 2: Concatenate and generate attention map.
        concat_features = torch.cat([p2_up, p3_up, p4], dim=1)  # (B, 3C, 15, 20)
        attn = self.attn_conv2(self.attn_relu(self.attn_conv1(concat_features)))  # (B, 1, 15, 20)
        attn = self.attn_sigmoid(attn)  # spatial attention map

        # Step 3: Upsample F4 and attention map to 60x80 for fine-grained enhancement.
        enhance_h, enhance_w = 60, 80
        f4_up = nn.functional.interpolate(
            f4, size=(enhance_h, enhance_w), mode="bilinear", align_corners=False
        )  # (B, 512, 60, 80)
        attn_up = nn.functional.interpolate(
            attn, size=(enhance_h, enhance_w), mode="bilinear", align_corners=False
        )  # (B, 1, 60, 80)

        # Step 4: Concatenate and enhance.
        concat_enhance = torch.cat([f4_up, attn_up], dim=1)  # (B, 513, 60, 80)
        enhanced = self.enhance_conv1(concat_enhance)
        enhanced = self.enhance_bn1(enhanced)
        enhanced = self.enhance_relu1(enhanced)
        enhanced = self.enhance_conv2(enhanced)
        enhanced = self.enhance_bn2(enhanced)
        enhanced = self.enhance_relu2(enhanced)  # (B, 512, 60, 80)

        # Step 5: Downsample back to 15x20.
        enhanced_down = nn.functional.interpolate(
            enhanced, size=(target_h, target_w), mode="bilinear", align_corners=False
        )  # (B, 512, 15, 20)

        return enhanced_down
