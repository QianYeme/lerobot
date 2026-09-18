import json
from pathlib import Path
import tempfile
import unittest

import torch

from lerobot.policies.act_det.water_keypoint import WaterPointLabels, water_keypoint_loss
from lerobot.policies.act_det.detection.mask_decoder import MaskDecoder
from lerobot.policies.act_det.configuration_act_det import ACTDetConfig
from lerobot.configs.policies import PreTrainedConfig


ROOT = Path(__file__).resolve().parents[3]
LABELS = ROOT / "outputs/water_center_pilot_20260918/keypoint_labels.json"


class WaterPointTests(unittest.TestCase):
    def setUp(self):
        self.labels = WaterPointLabels(LABELS)

    def test_four_reviewed_frames_and_unannotated(self):
        keys = [(20, f, "top") for f in [0, 205, 538, 717, 206]]
        target, valid = self.labels.heatmaps(keys, (60, 80))
        self.assertEqual(valid.tolist(), [True, False, True, True, False])
        self.assertIsNone(self.labels.labels[(20, 205, "top")]["point_xy"])
        self.assertNotIn((20, 206, "top"), self.labels.labels)
        for i in [0, 2, 3]:
            point = self.labels.labels[keys[i]]["point_xy"]
            peak = int(target[i].argmax())
            self.assertLessEqual(abs(peak % 80 - ((point[0]+.5)/8-.5)), .5)
            self.assertLessEqual(abs(peak // 80 - ((point[1]+.5)/8-.5)), .5)

    def test_occluded_nan_cannot_change_loss_or_gradient(self):
        target, valid = self.labels.heatmaps([(20, 0, "top"), (20, 205, "top")], (60, 80))
        logits = torch.zeros_like(target, requires_grad=True)
        expected = water_keypoint_loss(logits[:1], target[:1], valid[:1])
        with torch.no_grad():
            logits[1] = float("nan")
            target[1] = float("nan")
        loss = water_keypoint_loss(logits, target, valid)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertGreater(logits.grad[0].abs().sum().item(), 0)
        self.assertEqual(logits.grad[1].abs().sum().item(), 0)

    def test_all_occluded_keeps_action_gradient(self):
        target, valid = self.labels.heatmaps([(20, 205, "top")], (60, 80))
        logits = torch.zeros_like(target, requires_grad=True)
        action = torch.tensor(2., requires_grad=True)
        aux = water_keypoint_loss(logits, target, valid)
        (action.square() + aux).backward()
        self.assertEqual(aux.item(), 0)
        self.assertEqual(logits.grad.abs().sum().item(), 0)
        self.assertEqual(action.grad.item(), 4)

    def test_invalid_annotations_rejected(self):
        source = json.loads(LABELS.read_text())
        for mutate in [
            lambda d: d["frames"][1].update(point_xy=[100, 200]),
            lambda d: d["frames"][0].update(point_xy=[640, 200]),
            lambda d: d["frames"].append(d["frames"][0]),
        ]:
            data = json.loads(json.dumps(source))
            mutate(data)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "labels.json"
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    WaterPointLabels(path)

    def test_existing_decoder_receives_visible_only_gradients(self):
        torch.manual_seed(1000)
        torch.set_num_threads(2)
        keys = [(20, f, "top") for f in [0, 205, 538, 717]]
        target, valid = self.labels.heatmaps(keys, (60, 80))
        decoder = MaskDecoder(fpn_channels=8, mid_channels=4, output_resolution=(60, 80))
        features = [torch.randn(4, 8, h, w, requires_grad=True)
                    for h, w in [(12, 16), (6, 8), (3, 4)]]
        logits = decoder(*features, return_logits=True)
        logits.retain_grad()
        loss = water_keypoint_loss(logits, target, valid)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(decoder.predict[2].weight.grad.abs().sum().item(), 0)
        self.assertEqual(logits.grad[1].abs().sum().item(), 0)
        for feature in features:
            self.assertGreater(feature.grad[valid].abs().sum().item(), 0)
            self.assertEqual(feature.grad[1].abs().sum().item(), 0)

    def test_optional_config_rejects_incompatible_supervision(self):
        self.assertFalse(ACTDetConfig().use_water_keypoint)
        valid = dict(use_water_keypoint=True, use_mask_guidance=False, aug_enable=False)
        ACTDetConfig(**valid)
        for change in [dict(use_mask_guidance=True), dict(use_detection=False),
                       dict(mask_feature_inject=True), dict(water_keypoint_weight=-1),
                       dict(aug_enable=True, aug_occlusion_enable=True),
                       dict(det_cameras={"observation.images.top": {"enable": False}})]:
            with self.assertRaises(ValueError):
                ACTDetConfig(**(valid | change))

    def test_optional_config_roundtrip_without_label_file(self):
        config = ACTDetConfig(use_water_keypoint=True, use_mask_guidance=False,
                              aug_enable=False, water_keypoint_labels=None)
        with tempfile.TemporaryDirectory() as folder:
            config.save_pretrained(folder)
            restored = PreTrainedConfig.from_pretrained(folder)
            self.assertTrue(restored.use_water_keypoint)
            self.assertIsNone(restored.water_keypoint_labels)


if __name__ == "__main__":
    unittest.main()
