import importlib.util
from pathlib import Path

import numpy as np
import pyarrow as pa


SCRIPT = Path(__file__).parents[2] / "scripts" / "prepare_formal3_reset_preroll_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_formal3_reset_preroll_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_table(length: int, episode: int = 0) -> pa.Table:
    values = np.arange(length * 2, dtype=np.float32).reshape(length, 2)
    return pa.table(
        {
            "action": pa.array(values.tolist(), type=pa.list_(pa.float32(), 2)),
            "observation.state": pa.array((values + 100).tolist(), type=pa.list_(pa.float32(), 2)),
            "timestamp": np.arange(length, dtype=np.float32) / 30,
            "frame_index": np.arange(length, dtype=np.int64),
            "episode_index": np.full(length, episode, dtype=np.int64),
            "index": np.arange(length, dtype=np.int64),
            "task_index": np.zeros(length, dtype=np.int64),
        }
    )


def test_plateau_prefix_uses_one_target_then_preserves_aligned_trimmed_rows():
    original = make_table(12)
    trimmed = original.slice(4, 6)

    result, prefix = MODULE.build_episode_table(original, trimmed, start=4, requested_prefix=3)

    assert prefix == 3
    assert result.num_rows == 9
    np.testing.assert_array_equal(result["observation.state"].slice(0, 3).to_pylist(), original["observation.state"].slice(0, 3).to_pylist())
    np.testing.assert_array_equal(result["action"].slice(0, 3).to_pylist(), [original["action"][4].as_py()] * 3)
    assert result.slice(3).select(["action", "observation.state"]).equals(trimmed.select(["action", "observation.state"]))


def test_prefix_is_capped_at_motion_start_to_avoid_source_overlap():
    original = make_table(12)
    trimmed = original.slice(2, 7)

    result, prefix = MODULE.build_episode_table(original, trimmed, start=2, requested_prefix=5)

    assert prefix == 2
    assert result.num_rows == 9
    assert result["observation.state"].slice(0, 2).to_pylist() == original["observation.state"].slice(0, 2).to_pylist()


def test_remap_xml_keeps_prefix_and_full_aligned_task(tmp_path):
    source = tmp_path / "source.xml"
    destination = tmp_path / "destination.xml"
    source.write_text(
        '<annotations><track id="0"><box frame="0"/><box frame="2"/>'
        '<box frame="4"/><box frame="5"/><box frame="8"/></track></annotations>',
        encoding="utf-8",
    )

    assert MODULE.remap_xml(source, destination, start=4, end=8, prefix=3) == 4
    root = MODULE.ET.parse(destination).getroot()
    assert [box.attrib["frame"] for box in root.findall(".//box")] == ["0", "2", "3", "4"]
