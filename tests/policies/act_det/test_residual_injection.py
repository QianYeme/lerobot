"""Residual injection must retain DET tokens and an exact zero-alpha fallback."""
import copy
import unittest
import tempfile
from pathlib import Path

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act_det.configuration_act_det import ACTDetConfig
from lerobot.policies.act_det.modeling_act_det import ACTDetPolicy
from lerobot.utils.constants import OBS_IMAGES


class ResidualInjectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(1000)
        self.config = ACTDetConfig(
            device="cpu", pretrained_backbone_weights=None,
            input_features={
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
                "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 128)),
            },
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
            dim_model=64, dim_feedforward=128, n_heads=4, n_encoder_layers=1,
            chunk_size=3, n_action_steps=1, use_vae=False, dropout=0,
            aug_enable=False, use_mask_guidance=False,
        )
        self.batch = {"observation.state": torch.rand(1, 8),
                      "observation.images.top": torch.rand(1, 3, 96, 128)}

    def policies(self, mode="residual"):
        baseline = ACTDetPolicy(self.config).eval()
        config = copy.deepcopy(self.config)
        config.fcos_feature_inject = True
        config.fcos_inject_mode = mode
        injected = ACTDetPolicy(config).eval()
        result = injected.load_state_dict(baseline.state_dict(), strict=False)
        self.assertEqual(result.unexpected_keys, [])
        self.assertTrue(all("inject" in key or "residual_alpha" in key for key in result.missing_keys))
        return baseline, injected

    def test_zero_alpha_exact_fallback(self):
        baseline, injected = self.policies()
        injected.model.fcos_residual_alpha.data.zero_()
        with torch.no_grad():
            expected = baseline.predict_action_chunk(self.batch)
            actual = injected.predict_action_chunk(self.batch)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_positive_alpha_changes_actions_and_receives_gradients(self):
        baseline, injected = self.policies()
        expected = baseline.predict_action_chunk(self.batch)
        model_batch = dict(self.batch)
        model_batch[OBS_IMAGES] = [self.batch["observation.images.top"]]
        actual = injected.model(model_batch)[0]
        self.assertGreater((actual - expected).abs().max().item(), 0)
        actual.square().mean().backward()
        self.assertGreater(injected.model.fcos_residual_alpha.grad.abs().item(), 0)
        gradients = [p.grad for name, p in injected.named_parameters() if "inject" in name and p.grad is not None]
        self.assertTrue(any(gradient.abs().sum() > 0 for gradient in gradients))

    def test_default_remains_tokens_and_invalid_settings_rejected(self):
        self.assertEqual(self.config.fcos_inject_mode, "tokens")
        self.policies("tokens")
        with self.assertRaises(ValueError):
            ACTDetConfig(fcos_inject_mode="residual", fcos_inject_levels=["p3", "p4"])
        with self.assertRaises(ValueError):
            ACTDetConfig(fcos_residual_alpha=-0.1)

    def test_checkpoint_reload_preserves_residual_actions(self):
        _, injected = self.policies()
        with tempfile.TemporaryDirectory() as directory:
            injected.save_pretrained(Path(directory))
            reloaded = ACTDetPolicy.from_pretrained(Path(directory))
            self.assertEqual(reloaded.config.fcos_inject_mode, "residual")
            torch.testing.assert_close(reloaded.predict_action_chunk(self.batch),
                                       injected.predict_action_chunk(self.batch), rtol=0, atol=0)

    def test_effective_alpha_is_bounded(self):
        _, injected = self.policies()
        with torch.no_grad():
            injected.model.fcos_residual_alpha.fill_(-1)
            below = injected.predict_action_chunk(self.batch)
            injected.model.fcos_residual_alpha.zero_()
            zero = injected.predict_action_chunk(self.batch)
            injected.model.fcos_residual_alpha.fill_(2)
            above = injected.predict_action_chunk(self.batch)
            injected.model.fcos_residual_alpha.fill_(1)
            one = injected.predict_action_chunk(self.batch)
        torch.testing.assert_close(below, zero, rtol=0, atol=0)
        torch.testing.assert_close(above, one, rtol=0, atol=0)

    def test_residual_retains_baseline_token_count(self):
        baseline, residual = self.policies()
        _, tokens = self.policies("tokens")
        counts = []
        for policy in (baseline, residual, tokens):
            handle = policy.model.encoder.register_forward_pre_hook(
                lambda module, inputs: counts.append(inputs[0].shape[0]))
            policy.predict_action_chunk(self.batch)
            handle.remove()
        self.assertEqual(counts[0], counts[1])
        self.assertEqual(counts[2] - counts[0], 12)


if __name__ == "__main__":
    unittest.main()
