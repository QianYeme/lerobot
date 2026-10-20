import importlib.util
from pathlib import Path

import numpy as np

spec=importlib.util.spec_from_file_location('prepare_f2_1_r1',Path(__file__).resolve().parents[2]/'scripts/prepare_f2_1_r1.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_history_causal_with_earliest_padding():
    x=np.arange(30)
    w=m.windows(x)
    assert w.shape==(30,15)
    np.testing.assert_array_equal(w[0],np.zeros(15))
    np.testing.assert_array_equal(w[14],np.arange(15))
    x[15:]=999
    np.testing.assert_array_equal(m.windows(x)[:15],w[:15])


def test_each_episode_restarts_history():
    m.windows(np.arange(30))
    np.testing.assert_array_equal(m.windows([70,71])[0],np.full(15,70))


def test_state_removes_both_feedback_channels_not_zeroes():
    names=m.FIELDS+['gripper.load','gripper.curr']
    x=np.arange(24).reshape(3,8).astype(float)
    state,load=m.extract(x,names)
    assert state.shape==(3,6)
    x[:,7]=999
    a,b=m.extract(x,names)
    np.testing.assert_array_equal(a,state)
    np.testing.assert_array_equal(b,load)
    x[:,6]=88
    np.testing.assert_array_equal(m.extract(x,names)[0],state)


def test_h0_only_and_frame_weighted():
    pred=np.zeros((3,100,6));target=np.zeros((3,6))
    pred[:,0,5]=[1,2,6]
    pred[:,1:,5]=999
    assert m.h0_gripper_l1(pred,target)==3
