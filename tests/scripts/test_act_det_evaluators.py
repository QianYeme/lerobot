import pytest
import torch

from lerobot.scripts.eval_detection_localization import (
    average_precision,
    decode_fcos,
    operating_point,
    top1_metrics,
)
from lerobot.scripts.eval_visual_sensitivity import _summarize_details


def test_visual_detail_summary_groups_and_error_increase():
    counts = {
        name: torch.ones((2, 3), dtype=torch.int64)
        for name in (
            "original_action_l1",
            "swapped_action_l1",
            "blank_action_l1",
            "swap_action_delta_l1",
            "blank_action_delta_l1",
        )
    }
    sums = {
        "original_action_l1": torch.ones((2, 3)),
        "swapped_action_l1": torch.full((2, 3), 3.0),
        "blank_action_l1": torch.full((2, 3), 2.0),
        "swap_action_delta_l1": torch.full((2, 3), 4.0),
        "blank_action_delta_l1": torch.full((2, 3), 5.0),
    }

    result = _summarize_details(sums, counts, ["joint_0", "joint_1", "gripper.pos"])

    assert result["group_indices"] == {"arm": [0, 1], "gripper": [2]}
    assert result["by_action_group"]["swap_error_increase"] == {
        "arm": pytest.approx(2.0),
        "gripper": pytest.approx(2.0),
    }
    assert result["by_horizon"]["blank_error_increase"] == pytest.approx([1.0, 1.0])


def test_detection_decode_and_metrics():
    predictions = decode_fcos(
        cls_logits=[torch.tensor([[[[10.0]]]])],
        reg_preds=[torch.ones((1, 4, 1, 1))],
        ctr_preds=[torch.tensor([[[[10.0]]]])],
        strides=[8],
        image_size=(16, 16),
        score_floor=0.001,
        nms_iou=0.6,
        pre_nms_topk=10,
        max_detections=10,
    )
    torch.testing.assert_close(predictions[0]["boxes"], torch.tensor([[0.0, 0.0, 12.0, 12.0]]))

    predictions_by_image = {
        0: predictions[0],
        1: {"boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]]), "scores": torch.tensor([0.8])},
    }
    ground_truth = {
        0: torch.tensor([[0.0, 0.0, 12.0, 12.0]]),
        1: torch.empty((0, 4)),
    }

    assert average_precision(predictions_by_image, ground_truth, 0.5, 0.001) == pytest.approx(1.0)
    point = operating_point(predictions_by_image, ground_truth, 0.5, 0.05)
    assert point["true_positive"] == 1
    assert point["false_positive"] == 1
    assert point["false_negative"] == 0
    assert point["precision"] == pytest.approx(0.5)
    assert point["recall"] == pytest.approx(1.0)
    top1 = top1_metrics(predictions_by_image, ground_truth, (16, 16))
    assert top1["mean_iou"] == pytest.approx(1.0)
    assert top1["recall_iou_50"] == pytest.approx(1.0)
