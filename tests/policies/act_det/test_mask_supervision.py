"""Regression tests for the real ACTDet MASK forward/loss path (unittest compatible)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act_det.configuration_act_det import ACTDetConfig
from lerobot.policies.act_det.mask_loader import MaskLoader
from lerobot.policies.act_det.modeling_act_det import ACTDetPolicy
from lerobot.policies.act_det.detection.mask_decoder import mask_supervision_loss


class MaskSupervisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(1000)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.mask_dir = self.root / "masks"
        (self.mask_dir / "top").mkdir(parents=True)
        masks = np.zeros((2, 96, 128), dtype=np.float32)
        masks[:, 24:60, 40:80] = 1
        np.savez_compressed(self.mask_dir / "top/episode_000.npz", masks=masks)
        config = ACTDetConfig(
            device="cpu",
            input_features={
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
                "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 128)),
            },
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
            pretrained_backbone_weights=None,
            dim_model=64,
            dim_feedforward=128,
            n_heads=4,
            n_encoder_layers=1,
            chunk_size=3,
            n_action_steps=1,
            use_vae=False,
            dropout=0,
            aug_enable=False,
            use_mask_guidance=True,
            mask_dir=str(self.mask_dir),
            mask_weight=0.1,
            fcos_feature_inject=False,
            mask_feature_inject=False,
        )
        self.policy = ACTDetPolicy(config)
        self.policy.model.mask_decoder.output_resolution = (96, 128)
        self.policy.train()
        self.image = torch.rand(1, 3, 96, 128)
        self.state = torch.rand(1, 8)

    def tearDown(self):
        self.temp.cleanup()

    def batch(self, size=1, frames=None):
        return {
            "observation.state": self.state.repeat(size, 1),
            "observation.images.top": self.image.repeat(size, 1, 1, 1),
            "episode_index": torch.zeros(size, dtype=torch.long),
            "frame_index": torch.tensor(frames if frames is not None else [0] * size),
            "action": torch.zeros(size, 3, 6),
            "action_is_pad": torch.zeros(size, 3, dtype=torch.bool),
        }

    def test_tensor_loss_matches_pixel_mean_and_trains_shared_features(self):
        predictions = []
        handle = self.policy.model.mask_decoder.register_forward_hook(
            lambda module, inputs, output: predictions.append(output)
        )
        _, logs = self.policy(self.batch())
        handle.remove()
        loss = self.policy.model.get_mask_loss()["mask_loss"]
        self.assertIsInstance(loss, torch.Tensor)
        target = torch.from_numpy(np.load(self.mask_dir / "top/episode_000.npz")["masks"][:1, None])
        torch.testing.assert_close(loss, F.l1_loss(predictions[0], target))
        self.assertIsInstance(logs["mask_loss"], float)
        loss.backward()
        for name in ("mask_decoder", "fpn", "backbone"):
            grads = [p.grad for p in getattr(self.policy.model, name).parameters() if p.grad is not None]
            self.assertTrue(grads, name)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads), name)
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0, name)
        optimizer_ids = {id(p) for group in self.policy.get_optim_params() for p in group["params"]}
        self.assertTrue(all(id(p) in optimizer_ids for p in self.policy.model.mask_decoder.parameters()))

    def test_repeating_batch_does_not_shrink_loss(self):
        with torch.no_grad():
            for loss_type in ("l1", "bce_dice"):
                self.policy.config.mask_loss_type = loss_type
                values = []
                for size in (1, 4, 8):
                    self.policy(self.batch(size))
                    values.append(float(self.policy.model.get_mask_loss()["mask_loss"]))
                np.testing.assert_allclose(values, [values[0]] * 3, rtol=1e-5, atol=1e-6)

    def test_missing_frame_and_directory_raise_in_auxiliary_forward(self):
        with self.assertRaises(FileNotFoundError):
            self.policy(self.batch(2, [0, 2]))
        self.policy.model.mask_loader = MaskLoader(str(self.root / "absent"), strict=True)
        with self.assertRaises(FileNotFoundError):
            self.policy(self.batch())

    def test_inference_needs_neither_masks_nor_decoder(self):
        self.policy.model.mask_loader = None
        with patch.object(self.policy.model.mask_decoder, "forward", side_effect=AssertionError("decoder called")):
            actions = self.policy.predict_action_chunk(self.batch())
        self.assertEqual(tuple(actions.shape), (1, 3, 6))
        self.assertTrue(torch.isfinite(actions).all())

    def test_zero_weight_has_zero_mask_decoder_gradient(self):
        self.policy.config.mask_weight = 0
        loss, _ = self.policy(self.batch())
        loss.backward()
        for p in self.policy.model.mask_decoder.parameters():
            self.assertTrue(p.grad is None or torch.count_nonzero(p.grad) == 0)
        self.assertGreater(float(self.policy.model.action_head.weight.grad.abs().sum()), 0)

    def test_explicit_invalid_frame_does_not_discard_valid_sample(self):
        masks = np.zeros((2, 96, 128), dtype=np.float32)
        masks[0, 24:60, 40:80] = 1
        np.savez_compressed(self.mask_dir / "top/episode_000.npz", masks=masks, valid=[True, False])
        with torch.no_grad():
            self.policy(self.batch())
            expected = float(self.policy.model.get_mask_loss()["mask_loss"])
            self.policy(self.batch(2, [0, 1]))
            actual = float(self.policy.model.get_mask_loss()["mask_loss"])
        self.assertAlmostEqual(actual, expected, places=6)

    def test_reload_preserves_predictions(self):
        self.policy.config.mask_loss_type = "bce_dice"
        with torch.no_grad():
            expected = self.policy.predict_action_chunk(self.batch())
        save_dir = self.root / "checkpoint"
        self.policy.save_pretrained(save_dir)
        loaded = ACTDetPolicy.from_pretrained(save_dir)
        self.assertEqual(loaded.config.mask_loss_type, "bce_dice")
        with torch.no_grad():
            actual = loaded.predict_action_chunk(self.batch())
        torch.testing.assert_close(actual, expected)

    def test_all_explicitly_invalid_frames_report_zero_coverage_not_nan(self):
        np.savez_compressed(self.mask_dir / "top/episode_000.npz", masks=np.zeros((2, 96, 128)), valid=[False, False])
        loss, logs = self.policy(self.batch())
        self.assertEqual(logs["mask_coverage"], 0)
        self.assertEqual(logs["mask_valid_pixels"], 0)
        self.assertEqual(logs["mask_loss"], 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for p in self.policy.model.mask_decoder.parameters():
            self.assertTrue(p.grad is None or torch.count_nonzero(p.grad) == 0)

    def test_bce_dice_uses_logits_and_trains_full_policy_mask_path(self):
        self.policy.config.mask_loss_type = "bce_dice"
        predictions = []
        handle = self.policy.model.mask_decoder.register_forward_hook(
            lambda module, inputs, output: predictions.append(output)
        )
        loss, logs = self.policy(self.batch())
        handle.remove()
        target = torch.from_numpy(np.load(self.mask_dir / "top/episode_000.npz")["masks"][:1, None])
        probability = predictions[0].sigmoid()
        expected_dice = (2 * (probability * target).sum() + 1e-6) / (probability.sum() + target.sum() + 1e-6)
        expected = F.binary_cross_entropy_with_logits(predictions[0], target) + 1 - expected_dice
        torch.testing.assert_close(self.policy.model.get_mask_loss()["mask_loss"], expected)
        self.assertIsInstance(logs["mask_loss"], float)
        loss.backward()
        self.assertGreater(float(self.policy.model.mask_decoder.predict[2].weight.grad.abs().sum()), 0)

    def test_bce_dice_has_finite_foreground_gradient_for_saturated_negative_logits(self):
        prediction = torch.full((1, 1, 4, 5), -100.0, requires_grad=True)
        target = torch.ones_like(prediction)
        loss = mask_supervision_loss(prediction, target, "bce_dice")
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue((prediction.grad < 0).all())


class StrictMaskLoaderTests(unittest.TestCase):
    def test_corrupt_out_of_range_values_and_missing_files_are_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "top"
            path.mkdir()
            np.savez_compressed(path / "episode_000.npz", masks=np.full((1, 4, 5), np.nan))
            (path / "episode_001.npz").write_bytes(b"not an npz")
            loader = MaskLoader(directory, strict=True)
            for episode in (0, 1):
                with self.assertRaises(ValueError):
                    loader.get_mask("observation.images.top", episode, 0)
            with self.assertRaises(FileNotFoundError):
                loader.get_mask("observation.images.top", 2, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
