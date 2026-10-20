import importlib.util
from pathlib import Path

import numpy as np

spec=importlib.util.spec_from_file_location('prepare_f2_3',Path(__file__).parents[2]/'scripts/prepare_f2_3.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_residual_sign_and_clip():
    np.testing.assert_array_equal(module.residual_target([10,0,3],[0,10,2],2),[2,-2,1])


def test_zero_residual_identity():
    nominal=np.array([1.,2.,3.])
    np.testing.assert_array_equal(nominal+module.residual_target(nominal,nominal,2),nominal)
