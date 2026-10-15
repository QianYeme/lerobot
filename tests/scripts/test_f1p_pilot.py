import numpy as np

from scripts.eval_f1p_pilot import expand_labels, paired_bootstrap, score


def test_interval_expansion_excludes_uncertain_and_unlabeled_frames():
    rows = [dict(episode='10', label='PRECONTACT', first_reliable_frame='0', last_reliable_frame='4', uncertain_start='5', uncertain_end='6'),
            dict(episode='10', label='CONTACT_SUPPORTED', first_reliable_frame='7', last_reliable_frame='9', uncertain_start='', uncertain_end='')]
    labels, unknown = expand_labels(rows, 10, 12)
    np.testing.assert_equal(labels, [0, 0, 0, 0, 0, -1, -1, 1, 1, 1, -1, -1])
    assert unknown.sum() == 2


def test_bootstrap_is_paired_episode_mean_not_frame_weighting():
    indices = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    result = paired_bootstrap([.1, -.1], indices)
    assert result['point_estimate'] == 0
    assert result['bootstrap_mean'] == 0
    assert result['positive_episodes'] == 1
    identical = paired_bootstrap([.03, .03], indices)
    np.testing.assert_allclose(identical['percentile_95_ci'], [.03, .03])


def test_balanced_accuracy_is_mean_of_recalls_not_frame_accuracy():
    y = np.r_[np.zeros(90, dtype=int), np.ones(10, dtype=int)]
    result = score(y, np.zeros(100, dtype=int))
    assert result['BA'] == .5
    assert result['per_class_recall'] == [1., 0.]
    assert result['evaluable']
