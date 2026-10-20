import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('analyze_f1rm', Path(__file__).resolve().parents[2] / 'scripts/analyze_f1rm.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_history_features_exclude_present_and_future_command():
    s, a = np.zeros((6,9)), np.zeros((6,6))
    s[:,5] = np.arange(6)**2
    a[:,5] = [1,3,6,10,15,21]
    before = m.features(s,a)
    a[2:,5] = 999
    np.testing.assert_array_equal(m.features(s,a)[0],before[0])
    np.testing.assert_array_equal(before[0],[4,3,3,2])


def test_greedy_threshold_without_replacement_and_tie_order():
    pairs = m.greedy(np.array([[0.],[0.],[2.]]),np.array([[0.],[0.],[0.],[2.75]]),.75)
    assert pairs == [(0,0,0.),(1,1,0.),(2,3,.75)]


def test_force_columns_do_not_enter_features():
    s, a = np.zeros((5,9)), np.zeros((5,6))
    x = m.features(s,a)
    s[:,6:8] = 1e6
    np.testing.assert_array_equal(x,m.features(s,a))


def test_smd_and_degenerate_cases():
    assert m.smd([1,2,3],[2,3,4]) == 1
    assert m.smd([1,1],[1,1]) == 0
    assert m.smd([1,1],[2,2]) is None
    assert m.smd([],[]) is None


def test_labels_preserve_unknown_and_reject_overlap():
    row = dict(trial_id='formal5_kind2_ep001',label='UNKNOWN',first_reliable_frame='91',
               last_reliable_frame='149',reviewer='kg',reason='1')
    m.validate_annotations([row],[row['trial_id']])
    with pytest.raises(ValueError):
        m.validate_annotations([row,row],[row['trial_id']])


def test_unknown_not_allowed_as_empty_contact():
    row = dict(trial_id='formal5_kind1_ep000',label='CONTACT',first_reliable_frame='1',
               last_reliable_frame='2',reviewer='kg',reason='1')
    with pytest.raises(ValueError):
        m.validate_annotations([row],[row['trial_id']])


def test_cluster_bootstrap_resamples_whole_trials_reproducibly():
    groups = [np.ones(10),np.full(20,3.)]
    a = m.cluster_bootstrap(groups)
    assert a == m.cluster_bootstrap(groups)
    assert a['ci95'] == [1.,3.]
    assert m.cluster_bootstrap([np.ones(10)])['ci95'] is None
