import numpy as np

from scripts.freeze_reviewed_phase_labels import labels_from_review


def review(*, reviewed=True):
    return {
        "reviewed": reviewed,
        "steps": {
            "closing": {"start_frame": 10, "end_frame": 19},
            "lift_start": {"frame": 25},
            "release": {"start_frame": 70, "end_frame": 79},
        },
    }


def test_reviewed_boundaries_map_to_four_phases_and_unknown_tail():
    labels, intervals = labels_from_review(100, review())
    assert intervals == [
        {"start": 0, "stop": 10, "phase": "approach"},
        {"start": 10, "stop": 25, "phase": "grasp"},
        {"start": 25, "stop": 70, "phase": "lift_place"},
        {"start": 70, "stop": 80, "phase": "release"},
        {"start": 80, "stop": 100, "phase": "unknown"},
    ]
    assert np.all(labels[10:25] == "grasp")
    assert labels[79] == "release"
    assert labels[80] == "unknown"


def test_unreviewed_episode_is_rejected():
    try:
        labels_from_review(100, review(reviewed=False))
    except ValueError as error:
        assert "not reviewed" in str(error)
    else:
        raise AssertionError("Unreviewed episode must be rejected")


def test_out_of_order_boundaries_are_rejected():
    value = review()
    value["steps"]["release"]["start_frame"] = 20
    try:
        labels_from_review(100, value)
    except ValueError as error:
        assert "out of order" in str(error)
    else:
        raise AssertionError("Out-of-order boundaries must be rejected")
