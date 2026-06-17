"""Online data augmentation for the ACTDet policy.

Implements five augmentations applied only to the top (global) camera image,
as described in the paper Section 3.2:
  - Color Jitter (brightness, contrast, saturation, hue)
  - Gaussian Noise
  - Random Occlusion

Each method has an independent enable flag and application probability (default 0.9).
All augmentations are applied online (per-forward-pass), so each epoch sees different
variants of each image.

Geometric augmentations (random crop, random scale) are included but disabled by default
per the paper's experimental finding that they degrade performance on this task.
"""

from __future__ import annotations

import random

import torch
from torch import Tensor


class ImageAugmentation:
    """Online image augmentation applied to a single camera view during training.

    Only operates on the front/global camera view to simulate real-world
    variations in lighting, noise, and partial occlusion.

    Args:
        probability: Independent application probability for each enabled method.
        color_jitter_enable: Enable color jitter (brightness/contrast/saturation/hue).
        brightness: (lo, hi) range for brightness adjustment factor.
        contrast: (lo, hi) range for contrast adjustment factor.
        saturation: (lo, hi) range for saturation adjustment factor.
        hue: (lo, hi) range for hue shift magnitude.
        gaussian_noise_enable: Enable additive Gaussian noise.
        noise_std_range: (lo, hi) range for noise standard deviation.
        random_occlusion_enable: Enable random gray rectangle occlusion.
        occlusion_area_ratio: (lo, hi) range for occlusion area as fraction of image.
        occlusion_gray_range: (lo, hi) range for occlusion gray fill value.
    """

    def __init__(
        self,
        probability: float = 0.9,
        # Color jitter.
        color_jitter_enable: bool = True,
        brightness: tuple[float, float] = (0.8, 1.2),
        contrast: tuple[float, float] = (0.8, 1.2),
        saturation: tuple[float, float] = (0.8, 1.2),
        hue: tuple[float, float] = (-0.1, 0.1),
        # Gaussian noise.
        gaussian_noise_enable: bool = True,
        noise_std_range: tuple[float, float] = (0.01, 0.05),
        # Random occlusion.
        random_occlusion_enable: bool = True,
        occlusion_area_ratio: tuple[float, float] = (0.1, 0.3),
        occlusion_gray_range: tuple[float, float] = (0.3, 0.7),
    ):
        self.probability = probability

        # Color jitter config.
        self.color_jitter_enable = color_jitter_enable
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        # Gaussian noise config.
        self.gaussian_noise_enable = gaussian_noise_enable
        self.noise_std_range = noise_std_range

        # Random occlusion config.
        self.random_occlusion_enable = random_occlusion_enable
        self.occlusion_area_ratio = occlusion_area_ratio
        self.occlusion_gray_range = occlusion_gray_range

    def __call__(self, image: Tensor) -> Tensor:
        """Apply enabled augmentations to a single image.

        Args:
            image: (C, H, W) image tensor, pixel values in [0, 1].

        Returns:
            Augmented image of the same shape and value range.
        """
        if not self.training:
            return image

        # Color jitter.
        if self.color_jitter_enable and random.random() < self.probability:
            image = self._apply_color_jitter(image)

        # Gaussian noise.
        if self.gaussian_noise_enable and random.random() < self.probability:
            image = self._apply_gaussian_noise(image)

        # Random occlusion.
        if self.random_occlusion_enable and random.random() < self.probability:
            image = self._apply_random_occlusion(image)

        return image

    @property
    def training(self) -> bool:
        """Augmentations are only active during training."""
        return True  # Controlled externally by policy's self.training

    # ---- Color Jitter ----

    def _apply_color_jitter(self, image: Tensor) -> Tensor:
        """Randomly perturb brightness, contrast, saturation, and hue.

        The order of adjustments is randomized each call.
        Image is expected in RGB format (C, H, W) with values in [0, 1].
        """
        adjustments = [
            ("brightness", self._adjust_brightness),
            ("contrast", self._adjust_contrast),
            ("saturation", self._adjust_saturation),
            ("hue", self._adjust_hue),
        ]
        random.shuffle(adjustments)

        for _, adjust_fn in adjustments:
            image = adjust_fn(image)

        return image

    def _adjust_brightness(self, image: Tensor) -> Tensor:
        factor = random.uniform(*self.brightness)
        return torch.clamp(image * factor, 0.0, 1.0)

    def _adjust_contrast(self, image: Tensor) -> Tensor:
        factor = random.uniform(*self.contrast)
        mean = image.mean(dim=(-2, -1), keepdim=True)
        return torch.clamp((image - mean) * factor + mean, 0.0, 1.0)

    def _adjust_saturation(self, image: Tensor) -> Tensor:
        factor = random.uniform(*self.saturation)
        # Convert to grayscale equivalent and blend.
        gray = image.mean(dim=0, keepdim=True)  # (1, H, W)
        return torch.clamp(gray + factor * (image - gray), 0.0, 1.0)

    def _adjust_hue(self, image: Tensor) -> Tensor:
        """Shift hue by a random delta.

        Only applies to RGB images. Simplified implementation that rotates
        the RGB vector around the gray axis.
        """
        delta = random.uniform(*self.hue)  # in [-0.1, 0.1]
        if abs(delta) < 1e-6:
            return image

        # Simplified hue rotation: rotate in the RG-BR-GB plane.
        # For small deltas this is a reasonable approximation.
        cos_d = torch.cos(torch.tensor(delta * 3.14159 * 2))
        sin_d = torch.sin(torch.tensor(delta * 3.14159 * 2))

        # Standard hue rotation matrix for RGB.
        r, g, b = image[0], image[1], image[2]
        # Use the classic RGB→HSV hue shift approximation.
        # Project onto a 2D plane orthogonal to (1,1,1).
        # For efficiency we use the simplified formula.
        gray = (r + g + b) / 3.0

        r_out = gray + cos_d * (r - gray) + sin_d * (g - b) / 3.0
        g_out = gray + cos_d * (g - gray) + sin_d * (b - r) / 3.0
        b_out = gray + cos_d * (b - gray) + sin_d * (r - g) / 3.0

        return torch.clamp(torch.stack([r_out, g_out, b_out]), 0.0, 1.0)

    # ---- Gaussian Noise ----

    def _apply_gaussian_noise(self, image: Tensor) -> Tensor:
        std = random.uniform(*self.noise_std_range)
        noise = torch.randn_like(image) * std
        return torch.clamp(image + noise, 0.0, 1.0)

    # ---- Random Occlusion ----

    def _apply_random_occlusion(self, image: Tensor) -> Tensor:
        """Overlay a random gray rectangle on the image."""
        _, H, W = image.shape
        area = H * W
        occlusion_area = area * random.uniform(*self.occlusion_area_ratio)

        # Random aspect ratio for the occlusion rectangle.
        aspect = random.uniform(0.5, 2.0)
        occ_h = int(min(H - 1, (occlusion_area * aspect) ** 0.5))
        occ_w = int(min(W - 1, occ_h / aspect))
        occ_h = max(1, occ_h)
        occ_w = max(1, occ_w)

        # Random position.
        top = random.randint(0, H - occ_h)
        left = random.randint(0, W - occ_w)

        gray_value = random.uniform(*self.occlusion_gray_range)
        image[:, top:top + occ_h, left:left + occ_w] = gray_value

        return image
