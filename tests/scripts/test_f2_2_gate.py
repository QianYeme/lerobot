import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('prepare_f2_2', Path(__file__).parents[2] / 'scripts/prepare_f2_2.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_gate_boundary_and_direction():
    np.testing.assert_array_equal(module.gate([0, 44, 45, 100], 44), [1, 1, 0, 0])


def test_gate_future_independence():
    first = np.array([20., 70., 30., 80.])
    changed = first.copy()
    changed[2:] = 0
    np.testing.assert_array_equal(module.gate(first, 44)[:2], module.gate(changed, 44)[:2])
