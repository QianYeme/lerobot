"""Time alignment checks for offline chunk replay and continuity metrics."""

import unittest

import numpy as np
import torch

from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
from lerobot.scripts.offline_eval_act_sequence import ensemble_replay, queue_replay, sequence_metrics


class SequenceMetricTests(unittest.TestCase):
    def test_online_ensemble_matches_direct_diagonal_for_long_episode(self):
        chunks = np.random.default_rng(1000).normal(size=(30, 7, 6)).astype(np.float32)
        for coefficient in (0, 0.01, 0.03, 0.1):
            ensemble = ACTTemporalEnsembler(coefficient, chunks.shape[1])
            online = np.stack([ensemble.update(torch.from_numpy(chunk[None]))[0].numpy() for chunk in chunks])
            np.testing.assert_allclose(online, ensemble_replay(chunks, coefficient), rtol=1e-5, atol=1e-6)

    def test_identical_absolute_targets_have_zero_conflict_and_replay_error(self):
        truth = np.arange(20 + 10, dtype=np.float32)[:, None] * np.ones((1, 6))
        chunks = np.stack([truth[t:t + 10] for t in range(20)])
        metrics = sequence_metrics(chunks, chunks, np.ones((20, 10), dtype=bool))
        for value in metrics["chunk_conflict"].values():
            self.assertEqual(value["mean"]["arm"], 0)
        for value in metrics["execution_replay"].values():
            self.assertLess(value["mae"]["arm"], 1e-6)
        np.testing.assert_array_equal(queue_replay(chunks, 5), truth[:20])

    def test_invalid_padded_horizons_do_not_enter_conflict_metric(self):
        chunks = np.zeros((3, 3, 6))
        chunks[0, 2] = 999
        valid = np.ones((3, 3), dtype=bool)
        valid[0, 2] = False
        result = sequence_metrics(chunks, np.zeros_like(chunks), valid)
        self.assertEqual(result["chunk_conflict"]["h1_2"]["pairs"], 3)
        self.assertEqual(result["chunk_conflict"]["h1_2"]["mean"]["arm"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
