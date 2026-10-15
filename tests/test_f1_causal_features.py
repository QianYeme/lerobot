"""Small CPU-only directional checks; no dataset or robot imports."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('f1', Path(__file__).parents[1] / 'scripts/audit_f1_signals.py')
f1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f1)


class F1CausalFeatures(unittest.TestCase):
    def setUp(self):
        self.state = np.arange(40 * 8, dtype=float).reshape(40, 8)
        self.action = np.arange(40 * 6, dtype=float).reshape(40, 6)

    def test_history_and_single_frame_template(self):
        indices, x, valid = f1.features(self.state, self.action, 1)
        self.assertEqual(indices[0], 2)
        np.testing.assert_equal(x[0, [0, 4, 8, 12]], [21, 8, 11, 6])
        np.testing.assert_equal(x[:, [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23]], 0)
        self.assertTrue(valid.all())

    def test_no_same_row_or_future_commands(self):
        for width in (1, 9, 15):
            indices, x, _ = f1.features(self.state, self.action, width)
            altered = self.action.copy()
            altered[25:] = 1e6
            _, y, _ = f1.features(self.state, altered, width)
            np.testing.assert_equal(x[indices <= 25], y[indices <= 25])
            state = self.state.copy()
            state[26:] = 1e7
            _, y, _ = f1.features(state, self.action, width)
            np.testing.assert_equal(x[indices <= 25], y[indices <= 25])

    def test_full_window_no_padding_and_common_nan_mask(self):
        indices, _, valid = f1.features(self.state, self.action, 15)
        self.assertEqual(indices[0], 16)
        self.assertEqual(len(indices), 24)
        self.assertTrue(valid.all())
        self.state[20, 7] = np.nan
        indices, _, valid = f1.features(self.state, self.action, 15)
        np.testing.assert_equal(indices[~valid], np.arange(20, 35))


if __name__ == '__main__':
    unittest.main()
