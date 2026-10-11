from pathlib import Path

from scripts.prepare_d0_collection_schedule import build_schedule, validate


def test_schedule_covers_every_stratum_once_per_round() -> None:
    records = build_schedule(20261008)
    validate(records)
    for round_index in range(1, 6):
        round_rows = [row for row in records if row["round"] == round_index]
        assert len({row["stratum"] for row in round_rows}) == 12


def test_schedule_is_deterministic_and_rounds_are_not_identical() -> None:
    first = build_schedule(20261008)
    second = build_schedule(20261008)
    assert first == second
    orders = [
        [row["stratum"] for row in first if row["round"] == round_index]
        for round_index in range(1, 6)
    ]
    assert len({tuple(order) for order in orders}) == 5


def test_record_loop_persists_sent_action() -> None:
    source = Path("src/lerobot/scripts/lerobot_record.py").read_text(encoding="utf-8")
    assert "sent_action = robot.send_action(robot_action_to_send)" in source
    assert "build_dataset_frame(dataset.features, sent_action, prefix=ACTION)" in source
    assert "build_dataset_frame(dataset.features, action_values, prefix=ACTION)" not in source
