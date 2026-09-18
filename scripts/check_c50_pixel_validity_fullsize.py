"""Real-data full-size gradient smoke; synthetic validity fixture, not label QA."""
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act_det.modeling_act_det import ACTDetPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.offline_eval_act_det import collate


def main():
    torch.set_num_threads(4)
    torch.manual_seed(1000)
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/mask_pixel_validity_20260918/fullsize_results.json"
    if output.exists():
        raise FileExistsError(output)
    dataset_root = next(root.glob("*/formal1_C50_nomaster"))
    checkpoint = root / "outputs/train/C50_NOMASTER_DET_smoke_s1000/checkpoints/000020/pretrained_model"
    with np.load(root / "outputs/mask_inject_phase1_20260917/mask_preview/masks/top/episode_000.npz") as archive:
        masks = archive["masks"][:2].copy()
    pixels = np.zeros_like(masks,dtype=bool)
    pixels[0,:16,:16] = True
    y,x = np.argwhere(masks[0] == 1)[0]
    pixels[0,y,x] = True
    with tempfile.TemporaryDirectory(prefix="c50_valid_pixels_") as directory:
        top = Path(directory)/"top"; top.mkdir()
        np.savez_compressed(top/"episode_000.npz",masks=masks,valid_pixels=pixels)
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.use_mask_guidance = True; cfg.mask_loss_type = "bce_dice"
        cfg.mask_dir = directory; cfg.mask_weight = 0.1; cfg.mask_cache_episodes = 1
        cfg.mask_feature_inject = False; cfg.fcos_feature_inject = False
        cfg.annotation_dir = str(dataset_root/"annotations"); cfg.aug_enable = False; cfg.device = "cuda"
        policy = ACTDetPolicy(cfg).cuda()
        pre,post = make_pre_post_processors(cfg,pretrained_path=str(checkpoint))
        dataset = LeRobotDataset("QYyyyyyyy/formal1_C50_nomaster",root=dataset_root,episodes=[0],
            delta_timestamps={"action":[i/30 for i in range(cfg.chunk_size)]},video_backend="pyav")
        batch = collate([dataset[0],dataset[1]])
        frame_indices = batch["frame_index"]
        batch = pre(batch); batch["frame_index"] = frame_indices.cuda()
        predictions = []
        def hook(module,inputs,value):
            value.retain_grad(); predictions.append(value)
        handle = policy.model.mask_decoder.register_forward_hook(hook)
        policy.train(); loss,logs = policy(batch); handle.remove()
        assert torch.isfinite(loss) and logs["mask_valid_pixels"] == 257
        policy.model.get_mask_loss()["mask_loss"].backward()
        valid = torch.from_numpy(pixels[:,None]).cuda()
        assert (predictions[0].grad[~valid] == 0).all()
        assert predictions[0].grad[valid].abs().sum() > 0
        norms = {}
        for name in ["mask_decoder","fpn","backbone"]:
            grads = [p.grad for p in getattr(policy.model,name).parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            norms[name] = sum(float(g.square().sum()) for g in grads)**0.5
            assert norms[name] > 0
        policy.zero_grad(set_to_none=True)
        with torch.no_grad():
            expected = policy.predict_action_chunk(batch)
            policy.model.mask_loader = None
            actual = policy.predict_action_chunk(batch)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            assert torch.isfinite(post(actual[:,0])).all()
        report = {"status":"passed","state_dim":8,"action_dim":6,"image_shape":[480,640],
                  "batch_size":2,"valid_pixels":257,"mask_coverage":logs["mask_coverage"],
                  "ignored_decoder_output_gradients_zero":True,"gradient_norms":norms,
                  "inference_without_masks_exact":True,
                  "warning":"Real images and processors, random policy, synthetic partial-validity fixture. Not training results or approved partial labels."}
        output.write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(json.dumps(report))


if __name__ == "__main__":
    main()
