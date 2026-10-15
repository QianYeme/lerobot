import numpy as np

from scripts.match_f1q_confound import balance, greedy_match


def test_nearest_without_replacement_and_smaller_frame_tie():
    x = np.array([[0.], [0.], [0.], [0.]])
    y = np.array([0, 0, 1, 1]); frames = np.array([20, 10, 50, 40])
    pairs = greedy_match(x, y, frames)
    assert [(a, b) for a, b, _ in pairs] == [(1, 3), (0, 2)]


def test_distance_threshold_not_relaxed_and_rejection_does_not_consume():
    pairs = greedy_match(np.array([[0.], [1.], [.75]]), np.array([0, 1, 1]), np.array([0, 1, 2]))
    assert pairs == [(0, 2, .75)]


def test_smd_zero_variance_handling():
    result = balance(np.ones((4, 4)), np.array([0, 0, 1, 1]))
    assert all(r['SMD'] == 0 and r['balance_pass'] for r in result)
    x = np.ones((4, 4)); x[2:] = 2
    assert all(r['SMD'] is None and not r['balance_pass'] for r in balance(x, np.array([0, 0, 1, 1])))
