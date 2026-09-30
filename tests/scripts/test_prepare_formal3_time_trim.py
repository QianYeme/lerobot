import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[2] / "scripts" / "prepare_formal3_time_trim.py"
SPEC = importlib.util.spec_from_file_location("prepare_formal3_time_trim", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_find_motion_start_requires_sustained_arm_departure():
    actions = np.zeros((20, 6), dtype=np.float32)
    actions[4, 0] = 8.0
    actions[7:, 2] = 6.0

    assert MODULE.find_motion_start(actions, threshold=5.0, sustain=3, baseline_frames=4) == 7


def test_trim_xml_filters_and_reindexes_boxes(tmp_path):
    source = tmp_path / "source.xml"
    destination = tmp_path / "destination.xml"
    source.write_text(
        '<annotations><track id="0"><box frame="1"/><box frame="3"/>'
        '<box frame="5"/><box frame="8"/></track></annotations>',
        encoding="utf-8",
    )

    assert MODULE.trim_xml(source, destination, start=3, end=8) == 2
    root = MODULE.ET.parse(destination).getroot()
    assert [box.attrib["frame"] for box in root.findall(".//box")] == ["0", "2"]
