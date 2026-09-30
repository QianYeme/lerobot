"""Real-batch smoke for the h0 action auxiliary weight (server, GPU).

Loads a preflight 2-step checkpoint, runs one real training batch through the
policy with the h0 flag on and off (identical weights), and asserts:
- loss_on - loss_off == (W - 1) * h0_l1_loss / chunk_size exactly; all other
  loss terms (gripper channel weighting, KLD, detection) cancel in the
  difference, so this proves the row weighting composes with them;
- h0_l1_loss is finite, positive, and absent when the flag is off;
- backward through the flagged policy gives finite non-zero gradients.
"""
import copy
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act_det.modeling_act_det import ACTDetPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.offline_eval_act_det import collate


def main() -> None:
    data_root = Path(sys.argv[1])
    checkpoint = Path(sys.argv[2])
    weight = float(sys.argv[3])

    on = ACTDetPolicy.from_pretrained(checkpoint)
    assert on.config.use_h0_action_aux is True, "flag not recorded in checkpoint config"
    assert on.config.h0_action_weight == weight, "weight mismatch in checkpoint config"

    off_cfg = copy.deepcopy(on.config)
    off_cfg.use_h0_action_aux = False
    off = ACTDetPolicy(off_cfg)
    # Direct construction leaves the model on CPU; move it to the device the
    # preprocessor (loaded from the checkpoint) will move the batch to.
    off.to(next(on.parameters()).device)
    off.load_state_dict(on.state_dict())

    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    # Load the SAVED pre-processor from the checkpoint so normalization matches
    # training exactly (ImageNet stats), same as the offline eval pipeline.
    preprocessor, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(checkpoint))

    fps = json.loads((data_root / "meta/info.json").read_text())["fps"]
    dataset = LeRobotDataset(
        "QYyyyyyyy/formal3_kind_merged_nomaster_reset_preroll_fit48",
        root=data_root, episodes=[0],
        delta_timestamps={"action": [i / fps for i in range(cfg.chunk_size)]},
        video_backend="pyav",
    )
    loader = iter(DataLoader(Subset(dataset, range(1)), batch_size=1, collate_fn=collate, num_workers=0))
    batch = next(loader)

    # The ACT pre-processor drops frame_index (needed for per-frame detection
    # annotations); save and restore it, same as the eval pipeline.
    frame_index = batch.get("frame_index")
    batch = preprocessor(batch)
    if frame_index is not None:
        batch["frame_index"] = frame_index

    on.train()
    off.train()
    # Seed identically before each forward so the dropout masks (dropout=0.1
    # in the training config) match and the two models see identical outputs;
    # only then does the row-weighting identity hold exactly.
    torch.manual_seed(1234)
    loss_off, dict_off = off.forward(dict(batch))
    torch.manual_seed(1234)
    loss_on, dict_on = on.forward(dict(batch))
    h0 = dict_on["h0_l1_loss"]
    assert "h0_l1_loss" not in dict_off
    assert math.isfinite(h0) and h0 > 0, f"h0_l1_loss = {h0}"
    expected = loss_off + (weight - 1.0) * h0 / on.config.chunk_size
    torch.testing.assert_close(loss_on, expected, rtol=1e-5, atol=1e-5)

    loss_on.backward()
    gradients = [p.grad for p in on.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients), "non-finite parameter gradients"
    assert any(g.abs().sum() > 0 for g in gradients), "all gradients are zero"
    print(f"W{weight:g}: real-batch h0 smoke ok (h0_l1={h0:.4f}, loss {loss_on.item():.4f})")


if __name__ == "__main__":
    main()
