"""Real-image integration smoke for sparse water points, not a training run."""
import json
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act_det.modeling_act_det import ACTDetPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.offline_eval_act_det import collate


def main():
    torch.set_num_threads(4)
    torch.manual_seed(1000)
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/water_center_pilot_20260918/integration_results.json"
    if output.exists():
        raise FileExistsError(output)
    dataset_root = next(root.glob("*/formal1_C50_nomaster"))
    checkpoint = root / "outputs/train/C50_NOMASTER_DET_smoke_s1000/checkpoints/000020/pretrained_model"
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.use_mask_guidance = False
    cfg.mask_feature_inject = False
    cfg.fcos_feature_inject = False
    cfg.use_water_keypoint = True
    cfg.water_keypoint_labels = str(root / "outputs/water_center_pilot_20260918/keypoint_labels.json")
    cfg.water_keypoint_weight = 0.1
    cfg.aug_enable = False
    cfg.annotation_dir = str(dataset_root / "annotations")
    cfg.device = "cuda"
    cfg.__post_init__()
    policy = ACTDetPolicy(cfg).cuda()
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(checkpoint))
    dataset = LeRobotDataset("QYyyyyyyy/formal1_C50_nomaster", root=dataset_root, episodes=[20],
        delta_timestamps={"action": [i/30 for i in range(cfg.chunk_size)]}, video_backend="pyav")

    def batch_for(frames):
        batch = collate([dataset[f] for f in frames])
        indices = {key: batch[key].clone() for key in ["episode_index", "frame_index"]}
        batch = pre(batch)
        batch.update({key: value.cuda() for key, value in indices.items()})
        assert batch["observation.state"].shape[-1] == 8
        assert batch["action"].shape[-1] == 6
        assert batch["frame_index"].tolist() == frames
        assert batch["episode_index"].tolist() == [20]*len(frames)
        return batch

    batch = batch_for([0, 205, 538, 717])
    predictions = []
    def hook(module, inputs, value):
        value.retain_grad()
        predictions.append(value)
    handle = policy.model.water_decoder.register_forward_hook(hook)
    policy.train()
    loss, logs = policy(batch)
    handle.remove()
    aux = policy.model._water_loss["water_keypoint_loss"]
    assert torch.isfinite(loss) and logs["water_visible_frames"] == 3 and logs["water_coverage"] == .75
    expected = logs["l1_loss"] + logs.get("kld_loss", 0)*cfg.kl_weight
    if policy.model.get_detection_loss() is not None:
        expected += float(policy.model.get_detection_loss()[0].detach())*cfg.det_weight
    expected += logs["water_keypoint_loss"]*cfg.water_keypoint_weight
    assert abs(float(loss.detach())-expected) < 1e-4
    aux.backward(retain_graph=True)
    assert predictions[0].grad[1].abs().sum() == 0
    assert predictions[0].grad[[0, 2, 3]].abs().sum() > 0
    norms = {}
    for name in ["water_decoder", "fpn", "backbone"]:
        grads = [p.grad for p in getattr(policy.model, name).parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        norms[name] = sum(float(g.square().sum()) for g in grads)**0.5
        assert norms[name] > 0
    optimizer_ids = {id(p) for group in policy.get_optim_params() for p in group["params"]}
    assert all(id(p) in optimizer_ids for p in policy.model.water_decoder.parameters())
    policy.zero_grad(set_to_none=True)
    loss.backward()
    assert policy.model.action_head.weight.grad.abs().sum() > 0
    policy.zero_grad(set_to_none=True)
    occluded = batch_for([205])
    occluded_loss, occluded_logs = policy(occluded)
    assert occluded_logs["water_keypoint_loss"] == 0 and occluded_logs["water_visible_frames"] == 0
    occluded_loss.backward()
    assert policy.model.action_head.weight.grad.abs().sum() > 0
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in policy.model.water_decoder.parameters())
    policy.zero_grad(set_to_none=True)
    with torch.no_grad():
        expected_actions = policy.predict_action_chunk(batch)
        policy.model.water_point_labels = None
        cfg.water_keypoint_labels = None
        deployment_batch = {k: v for k, v in batch.items() if k not in ["episode_index", "frame_index", "action", "action_is_pad"]}
        with patch.object(policy.model.water_decoder, "forward", side_effect=AssertionError("aux decoder called in deployment")):
            actual = policy.predict_action_chunk(deployment_batch)
        torch.testing.assert_close(actual, expected_actions, rtol=0, atol=0)
        assert torch.isfinite(post(actual[:, 0])).all()
        assert policy.model._water_loss is None
    policy.train()
    try:
        policy(occluded)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Missing auxiliary labels silently accepted")
    legacy_config = PreTrainedConfig.from_pretrained(checkpoint)
    legacy_policy = ACTDetPolicy(legacy_config)
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as saved:
        assert set(legacy_policy.state_dict()) == set(saved.keys())
    report = {
        "status": "passed", "frames": [0, 205, 538, 717], "visible_frames": 3,
        "state_dim": 8, "action_dim": 6, "image_size_wh": [640, 480],
        "cameras": list(cfg.image_features), "heatmap_size_hw": [60, 80],
        "gradient_norms_aux_only": norms, "ignored_output_gradient_zero": True,
        "weighted_joint_loss_verified": True, "occluded_frame_action_gradient_nonzero": True,
        "inference_without_labels_metadata_or_aux_decoder_exact": True,
        "missing_labels_rejected_for_training": True,
        "default_parameter_keys_match_old_checkpoint": True,
        "warning": "Random policy, real images and saved processors; no optimizer steps, no localization quality claim."
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
