"""Default-off h0 action auxiliary weight: legacy loss unchanged when off.

The h0 auxiliary weight up-weights the first chunk row in the action L1 loss
because n_action_steps=1 deployment executes h0 only. These tests pin:
- default off / positive weight validation;
- flag off (or weight=1.0) is bit-identical to the legacy loss;
- W>1 raises the loss by exactly (W-1)*h0_l1_loss/chunk_size;
- the monitored h0_l1_loss matches the manual row-0 L1;
- checkpoint reload preserves the flag and the weighted loss.
"""
import copy
import unittest
import tempfile
from pathlib import Path

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act_det.configuration_act_det import ACTDetConfig
from lerobot.policies.act_det.modeling_act_det import ACTDetPolicy
from lerobot.utils.constants import ACTION


class H0ActionAuxTests(unittest.TestCase):
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
                      "observation.images.top": torch.rand(1, 3, 96, 128),
                      ACTION: torch.rand(1, 3, 6),
                      "action_is_pad": torch.zeros(1, 3, dtype=torch.bool)}

    def policies(self, weight=1.0):
        off = ACTDetPolicy(self.config).eval()
        config = copy.deepcopy(self.config)
        config.use_h0_action_aux = True
        config.h0_action_weight = weight
        on = ACTDetPolicy(config).eval()
        on.load_state_dict(off.state_dict())
        return off, on

    def test_default_off_and_validation(self):
        self.assertFalse(ACTDetConfig().use_h0_action_aux)
        self.assertEqual(ACTDetConfig().h0_action_weight, 1.0)
        with self.assertRaises(ValueError):
            ACTDetConfig(h0_action_weight=0.0)

    def test_off_and_weight_one_are_bit_identical(self):
        off, on = self.policies(weight=1.0)
        loss_off, dict_off = off.forward(dict(self.batch))
        loss_on, dict_on = on.forward(dict(self.batch))
        torch.testing.assert_close(loss_on, loss_off, rtol=0, atol=0)
        self.assertEqual(dict_on["l1_loss"], dict_off["l1_loss"])
        self.assertNotIn("h0_l1_loss", dict_off)
        self.assertIn("h0_l1_loss", dict_on)
        torch.testing.assert_close(on.predict_action_chunk(dict(self.batch)),
                                   off.predict_action_chunk(dict(self.batch)), rtol=0, atol=0)

    def test_weight_scales_loss_by_exact_row0_amount(self):
        off, on = self.policies(weight=5.0)
        loss_off, _ = off.forward(dict(self.batch))
        loss_on, dict_on = on.forward(dict(self.batch))
        expected = loss_off + (5.0 - 1.0) * dict_on["h0_l1_loss"] / self.config.chunk_size
        self.assertAlmostEqual(loss_on.item(), expected.item(), places=6)

    def test_h0_monitor_matches_manual_row0_l1(self):
        _, on = self.policies(weight=3.0)
        with torch.no_grad():
            actions_hat = on.predict_action_chunk(dict(self.batch))
        manual = torch.nn.functional.l1_loss(self.batch[ACTION][:, 0], actions_hat[:, 0]).item()
        _, dict_on = on.forward(dict(self.batch))
        self.assertAlmostEqual(dict_on["h0_l1_loss"], manual, places=6)

    def test_checkpoint_reload_preserves_flag(self):
        _, on = self.policies(weight=4.0)
        loss_on, _ = on.forward(dict(self.batch))
        with tempfile.TemporaryDirectory() as directory:
            on.save_pretrained(Path(directory))
            reloaded = ACTDetPolicy.from_pretrained(Path(directory))
            self.assertTrue(reloaded.config.use_h0_action_aux)
            self.assertEqual(reloaded.config.h0_action_weight, 4.0)
            loss_reloaded, dict_reloaded = reloaded.forward(dict(self.batch))
        torch.testing.assert_close(loss_reloaded, loss_on, rtol=0, atol=0)
        self.assertIn("h0_l1_loss", dict_reloaded)


if __name__ == "__main__":
    unittest.main()
