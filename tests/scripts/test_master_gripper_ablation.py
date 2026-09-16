import json

import pytest
import torch

from lerobot.scripts.eval_master_gripper_ablation import load_phases, summarize_h0, zero_master_state


@pytest.mark.parametrize("shape", [(2, 9), (2, 1, 9)])
def test_raw_zero_preserves_original_and_other_fields(shape):
    state = torch.arange(torch.tensor(shape).prod()).float().reshape(shape)
    original = state.clone()
    batch = {"observation.state": state, "action": torch.ones(2, 10, 6)}
    zero = zero_master_state(batch, 8)
    torch.testing.assert_close(batch["observation.state"], original)
    torch.testing.assert_close(zero["observation.state"][..., :8], original[..., :8])
    assert torch.count_nonzero(zero["observation.state"][..., 8]) == 0
    assert zero["action"] is batch["action"]
    # Raw zero becomes -2 after this synthetic normalizer, not normalized zero.
    assert torch.all((zero["observation.state"][..., 8] - 20) / 10 == -2)


def test_h0_summary_grouping_and_delta():
    target = torch.zeros(2, 6)
    original = torch.ones(2, 6)
    zero = torch.full((2, 6), 2.0)
    zero[:, 5] = 4
    names = [f"joint_{i}.pos" for i in range(5)] + ["gripper.pos"]
    result = summarize_h0(original, zero, target, names)
    assert result["frames"] == 2
    assert result["by_group"]["arm"]["error_increase"] == pytest.approx(1)
    assert result["by_group"]["gripper"]["error_increase"] == pytest.approx(3)
    assert result["by_group"]["all_unweighted"]["zero_master_l1"] == pytest.approx(14 / 6)
    assert summarize_h0(original[:0], zero[:0], target[:0], names) == {"frames": 0}


def test_phase_intervals_require_review_and_preserve_unknown_gaps(tmp_path):
    path = tmp_path / "phases.json"
    annotations = {"episodes": {"7": {"reviewed": True, "segments": [
        {"start": 0, "stop": 2, "phase": "approach"},
        {"start": 3, "stop": 5, "phase": "place"},
    ]}}}
    path.write_text(json.dumps(annotations), encoding="utf-8")
    phases = load_phases(path, {7: 5})
    assert phases[(7, 0)] == "approach"
    assert phases[(7, 3)] == "place"
    assert phases.get((7, 2), "unknown") == "unknown"
    annotations["episodes"]["7"]["reviewed"] = False
    path.write_text(json.dumps(annotations), encoding="utf-8")
    with pytest.raises(ValueError, match="unreviewed"):
        load_phases(path, {7: 5})
    annotations["episodes"]["7"]["reviewed"] = True
    annotations["episodes"]["7"]["segments"][1]["start"] = 1
    path.write_text(json.dumps(annotations), encoding="utf-8")
    with pytest.raises(ValueError, match="Overlapping"):
        load_phases(path, {7: 5})
