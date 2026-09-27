from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from lerobot.policies.act_det.configuration_act_det import ACTDetConfig
from lerobot.policies.act_det.phase_supervision import PhaseLabels, phase_classification_loss


class PhaseSupervisionTests(unittest.TestCase):
    def test_default_off_and_validation(self):
        self.assertFalse(ACTDetConfig().use_phase_aux)
        with self.assertRaises(ValueError):
            ACTDetConfig(phase_num_classes=5)
        with self.assertRaises(ValueError):
            ACTDetConfig(phase_weight=-0.1)

    def test_loader_masks_unknown_and_missing_rows(self):
        rows = [
            {"episode_index": 0, "frame_index": 0, "phase_id": 2, "valid": True},
            {"episode_index": 0, "frame_index": 1, "phase_id": -1, "valid": False},
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "labels.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path)
            labels = PhaseLabels(path)
            targets = labels.targets(torch.tensor([0, 0, 9]), torch.tensor([0, 1, 9]), device=torch.device("cpu"))
        self.assertEqual(targets.tolist(), [2, -1, -1])

    def test_unknown_has_zero_gradient(self):
        logits = torch.zeros(2, 4, requires_grad=True)
        loss = phase_classification_loss(logits, torch.tensor([1, -1]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(logits.grad[0].abs().sum().item(), 0)
        self.assertEqual(logits.grad[1].abs().sum().item(), 0)

    def test_all_unknown_is_finite_zero(self):
        logits = torch.randn(2, 4, requires_grad=True)
        loss = phase_classification_loss(logits, torch.tensor([-1, -1]))
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertEqual(logits.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
