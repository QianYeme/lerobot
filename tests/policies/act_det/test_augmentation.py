import torch

from lerobot.policies.act_det.detection.augmentation import ImageAugmentation


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def test_normalized_image_is_augmented_in_pixel_space() -> None:
    pixel_image = torch.full((3, 4, 5), 0.8)
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    normalized_image = (pixel_image - mean) / std
    augmentation = ImageAugmentation(
        probability=1.0,
        brightness=(0.5, 0.5),
        contrast=(1.0, 1.0),
        saturation=(1.0, 1.0),
        hue=(0.0, 0.0),
        gaussian_noise_enable=False,
        random_occlusion_enable=False,
    )

    result = augmentation(normalized_image, normalization_mean=MEAN, normalization_std=STD)

    expected = (torch.full_like(pixel_image, 0.4) - mean) / std
    torch.testing.assert_close(result, expected)


def test_normalization_parameters_must_be_paired() -> None:
    augmentation = ImageAugmentation()

    try:
        augmentation(torch.zeros(3, 2, 2), normalization_mean=MEAN)
    except ValueError as error:
        assert "provided together" in str(error)
    else:
        raise AssertionError("Expected mismatched normalization parameters to fail")
