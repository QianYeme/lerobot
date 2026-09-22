"""C0 explicit box conditioning must be deployable and gradient-isolated."""

import tempfile
import unittest
from pathlib import Path

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act_det.configuration_act_det import ACTDetConfig
from lerobot.policies.act_det.detection.fcos import decode_fcos_top1_condition
from lerobot.policies.act_det.modeling_act_det import ACTDetPolicy
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


class ExplicitBoxConditionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(1000)
        self.base_kwargs = dict(
            device="cpu",
            pretrained_backbone_weights=None,
            input_features={
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
                "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 128)),
            },
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
            dim_model=64,
            dim_feedforward=128,
            n_heads=4,
            n_encoder_layers=1,
            chunk_size=3,
            n_action_steps=1,
            use_vae=False,
            dropout=0,
            aug_enable=False,
            use_mask_guidance=False,
        )
        self.batch = {
            "observation.state": torch.rand(1, 8),
            "observation.images.top": torch.rand(1, 3, 96, 128),
        }

    def make_policy(self, enabled: bool, mode: str = "token") -> ACTDetPolicy:
        config = ACTDetConfig(
            **self.base_kwargs,
            use_explicit_box_condition=enabled,
            box_condition_mode=mode,
            box_condition_score_threshold=0,
        )
        return ACTDetPolicy(config)

    def test_decoder_normalizes_visible_box_and_masks_missing_prediction(self):
        cls = [torch.tensor([[[[8.0, -8.0]]]])]
        ctr = [torch.tensor([[[[8.0, -8.0]]]])]
        reg = [torch.tensor([[[[0.5, 0.5]], [[0.5, 0.5]], [[0.5, 0.5]], [[0.5, 0.5]]]])]
        condition = decode_fcos_top1_condition(cls, reg, ctr, [8], (16, 16), 0.25)
        torch.testing.assert_close(condition[0, :4], torch.tensor([-0.5, -0.5, 0.5, 0.5]))
        self.assertGreater(condition[0, 4].item(), 0.99)
        self.assertEqual(condition[0, 5].item(), 1)

        missing = decode_fcos_top1_condition(cls, reg, ctr, [8], (16, 16), 1.0)
        torch.testing.assert_close(missing, torch.zeros_like(missing))

    def test_c0_adds_exactly_one_encoder_token(self):
        baseline = self.make_policy(False).eval()
        conditioned = self.make_policy(True).eval()
        counts = []
        for policy in (baseline, conditioned):
            handle = policy.model.encoder.register_forward_pre_hook(
                lambda module, inputs: counts.append(inputs[0].shape[0])
            )
            policy.predict_action_chunk(self.batch)
            handle.remove()
        self.assertEqual(counts[1] - counts[0], 1)

    def test_condition_stays_a_1d_token_when_top_is_second_camera(self):
        kwargs = dict(self.base_kwargs)
        kwargs["input_features"] = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            "observation.images.gripper": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 128)),
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 128)),
        }
        policy = ACTDetPolicy(
            ACTDetConfig(
                **kwargs,
                use_explicit_box_condition=True,
                box_condition_score_threshold=0,
            )
        ).eval()
        batch = dict(self.batch)
        batch["observation.images.gripper"] = torch.rand(1, 3, 96, 128)
        captured = {}

        def capture_encoder_inputs(module, args, hook_kwargs):
            captured["tokens"] = args[0].detach().clone()
            captured["positions"] = hook_kwargs["pos_embed"].detach().clone()

        handle = policy.model.encoder.register_forward_pre_hook(capture_encoder_inputs, with_kwargs=True)
        policy.predict_action_chunk(batch)
        handle.remove()

        condition = policy.model.get_box_condition()
        expected_token = policy.model.box_condition_input_proj(condition)
        torch.testing.assert_close(captured["tokens"][2], expected_token)
        torch.testing.assert_close(
            captured["positions"][2, 0], policy.model.encoder_1d_feature_pos_embed.weight[2]
        )

    def test_c1_fuses_box_into_existing_state_token_without_adding_a_token(self):
        baseline = self.make_policy(False).eval()
        conditioned = self.make_policy(True, "state").eval()
        captured = {}

        def capture(name):
            return lambda _module, inputs: captured.setdefault(name, inputs[0].detach().clone())

        baseline_handle = baseline.model.encoder.register_forward_pre_hook(capture("baseline"))
        conditioned_handle = conditioned.model.encoder.register_forward_pre_hook(capture("conditioned"))
        baseline.predict_action_chunk(self.batch)
        conditioned.predict_action_chunk(self.batch)
        baseline_handle.remove()
        conditioned_handle.remove()

        self.assertEqual(captured["baseline"].shape[0], captured["conditioned"].shape[0])
        condition = conditioned.model.get_box_condition()
        expected = conditioned.model.encoder_robot_state_input_proj(self.batch[OBS_STATE])
        expected = expected + conditioned.model.box_condition_state_proj(condition)
        torch.testing.assert_close(captured["conditioned"][1], expected)

    def test_c2_adds_a_sigmoid_gated_action_residual(self):
        policy = self.make_policy(True, "action_residual").eval()
        normal = policy.predict_action_chunk(self.batch)
        residual, alpha = policy.model.get_box_action_residual()
        self.assertIsNotNone(residual)
        self.assertAlmostEqual(float(alpha.detach()), 0.05, places=6)

        handle = policy.model.box_condition_action_residual.register_forward_hook(
            lambda _module, _inputs, output: torch.zeros_like(output)
        )
        base = policy.predict_action_chunk(self.batch)
        handle.remove()
        torch.testing.assert_close(normal, base + alpha * residual)

    def test_shared_initialization_is_identical_across_modes(self):
        policies = []
        for mode in ("token", "state", "action_residual"):
            torch.manual_seed(123)
            policies.append(self.make_policy(True, mode))
        state_dicts = [policy.state_dict() for policy in policies]
        common_keys = set.intersection(*(set(state) for state in state_dicts))
        compared = 0
        for key in sorted(common_keys):
            values = [state[key] for state in state_dicts]
            if key.startswith("model.box_condition_") or len({tuple(value.shape) for value in values}) != 1:
                continue
            for value in values[1:]:
                torch.testing.assert_close(value, values[0], rtol=0, atol=0, msg=lambda message: f"{key}: {message}")
            compared += 1
        self.assertGreater(compared, 50)

    def test_action_gradient_stops_before_fcos_predictions(self):
        modules = {
            "token": "box_condition_input_proj",
            "state": "box_condition_state_proj",
            "action_residual": "box_condition_action_residual",
        }
        for mode, module_name in modules.items():
            with self.subTest(mode=mode):
                policy = self.make_policy(True, mode).train()
                model_batch = dict(self.batch)
                model_batch[OBS_IMAGES] = [self.batch["observation.images.top"]]
                actions = policy.model(model_batch, compute_aux_losses=False)[0]
                actions.square().mean().backward()

                condition_grads = [
                    parameter.grad
                    for parameter in getattr(policy.model, module_name).parameters()
                    if parameter.grad is not None
                ]
                self.assertTrue(any(gradient.abs().sum() > 0 for gradient in condition_grads))
                fcos_output_grads = [
                    parameter.grad
                    for name, parameter in policy.model.fcos_head.named_parameters()
                    if name.startswith(("cls_logits", "reg_pred", "ctr_pred"))
                ]
                self.assertTrue(all(gradient is None for gradient in fcos_output_grads))

    def test_eval_exposes_detached_condition_and_checkpoint_roundtrip(self):
        for mode in ("token", "state", "action_residual"):
            with self.subTest(mode=mode):
                policy = self.make_policy(True, mode).eval()
                expected = policy.predict_action_chunk(self.batch)
                condition = policy.model.get_box_condition()
                self.assertIsNotNone(condition)
                self.assertFalse(condition.requires_grad)
                self.assertEqual(tuple(condition.shape), (1, 6))

                with tempfile.TemporaryDirectory() as directory:
                    policy.save_pretrained(Path(directory))
                    reloaded = ACTDetPolicy.from_pretrained(Path(directory))
                    self.assertTrue(reloaded.config.use_explicit_box_condition)
                    self.assertEqual(reloaded.config.box_condition_mode, mode)
                    torch.testing.assert_close(
                        reloaded.predict_action_chunk(self.batch), expected, rtol=0, atol=0
                    )

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            ACTDetConfig(**self.base_kwargs, use_detection=False, use_explicit_box_condition=True)
        with self.assertRaises(ValueError):
            ACTDetConfig(**self.base_kwargs, box_condition_dropout=1.1)
        with self.assertRaises(ValueError):
            ACTDetConfig(**self.base_kwargs, box_condition_noise_std=-0.1)
        with self.assertRaises(ValueError):
            ACTDetConfig(**self.base_kwargs, box_condition_mode="unknown")
        with self.assertRaises(ValueError):
            ACTDetConfig(**self.base_kwargs, box_action_residual_alpha=0.0)
        with self.assertRaises(ValueError):
            ACTDetConfig(**self.base_kwargs, box_action_residual_alpha=1.0)
        missing_camera = ACTDetConfig(
            **self.base_kwargs,
            use_explicit_box_condition=True,
            box_condition_camera="observation.images.missing",
        )
        with self.assertRaises(ValueError):
            ACTDetPolicy(missing_camera)


if __name__ == "__main__":
    unittest.main()
