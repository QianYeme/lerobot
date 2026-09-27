import numpy as np

from scripts.export_formal3_phase_labels import find_events, label_episode


def make_normal_episode():
    gripper = np.zeros(100, dtype=np.float32)
    gripper[10:30] = 40
    gripper[30:70] = 20
    gripper[70:90] = 40
    gripper[90:] = 20
    load = np.zeros(100, dtype=np.float32)
    load[30:70] = 80
    return gripper, load


def detect(gripper, load):
    return find_events(
        gripper,
        gripper,
        load,
        open_threshold=30,
        close_threshold=25,
        sustain_frames=3,
        baseline_frames=60,
        contact_delta=40,
        motion_margin=0.1,
    )


def test_normal_sequence_has_four_phases_and_unknown_reset():
    events = detect(*make_normal_episode())
    labels, intervals, warnings = label_episode(100, events, hold_frames=9, contact_tolerance=15)

    assert events == {
        "open": 10,
        "close": 30,
        "contact": 30,
        "actual_close": 30,
        "close_motion_start": 30,
        "close_motion_end": 30,
        "release_motion_start": 70,
        "release_motion_end": 70,
        "release": 70,
        "reset_close": 90,
        "load_baseline": 0.0,
        "contact_threshold": 40.0,
    }
    assert warnings == []
    assert intervals == [
        {"start": 0, "stop": 30, "phase": "approach"},
        {"start": 30, "stop": 31, "phase": "grasp"},
        {"start": 31, "stop": 70, "phase": "lift_place"},
        {"start": 70, "stop": 90, "phase": "release"},
        {"start": 90, "stop": 100, "phase": "unknown"},
    ]
    assert len(labels) == 100


def test_delayed_load_contact_is_diagnostic_only():
    gripper, load = make_normal_episode()
    load[:] = 0
    load[55:70] = 80
    events = detect(gripper, load)
    labels, _, warnings = label_episode(100, events, hold_frames=9, contact_tolerance=15)

    assert events["contact"] == 55
    assert warnings == ["load_contact_mismatch_diagnostic"]
    assert labels[30] == "grasp"
    assert np.all(labels[31:70] == "lift_place")
    assert np.all(labels[:30] == "approach")
    assert np.all(labels[70:90] == "release")


def test_missing_reset_fails_closed():
    gripper, load = make_normal_episode()
    gripper[90:] = 40
    events = detect(gripper, load)
    labels, intervals, warnings = label_episode(100, events, hold_frames=9, contact_tolerance=15)

    assert events["reset_close"] is None
    assert intervals == []
    assert warnings == ["missing_command_event"]
    assert np.all(labels == "unknown")
