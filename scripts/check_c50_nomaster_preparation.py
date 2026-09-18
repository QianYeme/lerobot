"""Verify smoke checkpoints, then probe three simultaneous batch-8 training workers."""
import csv
import hashlib
import json
import math
import multiprocessing as mp
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file
from torch.utils.data import DataLoader

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.offline_eval_act_det import POLICY_CLASSES, collate

ROOT = Path("outputs/formal1_C50_nomaster_phase1")
DATA = Path("数据集/formal1_C50_nomaster")
MODELS = {"ACT": ("act", False), "DET": ("act_det", False), "INJECT": ("act_det", True)}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def worker(model, barrier):
    name = f"C50_NOMASTER_{model}_smoke_s1000"
    checkpoint = Path("outputs/train") / name / "checkpoints/000020/pretrained_model"
    manifest = json.loads((DATA / "meta/nomaster_manifest.json").read_text())
    stats = json.loads((DATA / "meta/stats.json").read_text())
    cfg = TrainPipelineConfig.from_pretrained(checkpoint)
    assert cfg.steps == 20 and not cfg.resume and cfg.policy.pretrained_path is None
    assert cfg.seed == 1000 and cfg.batch_size == 8 and cfg.num_workers == 4
    assert cfg.dataset.episodes == manifest["train"]
    assert cfg.policy.type == MODELS[model][0] and not cfg.policy.use_amp
    assert cfg.policy.input_features["observation.state"].shape == (8,)
    assert cfg.policy.output_features["action"].shape == (6,)
    assert cfg.policy.chunk_size == 100 and cfg.policy.n_action_steps == 1
    assert cfg.policy.gripper_loss_weight == 3 and cfg.policy.optimizer_lr_backbone == 1e-4
    if model != "ACT":
        assert cfg.policy.use_detection and cfg.policy.fcos_feature_inject == MODELS[model][1]
        assert not cfg.policy.use_mask_guidance and not cfg.policy.aug_enable
    with (Path("outputs/train") / name / "metrics.csv").open() as stream:
        metrics = list(csv.DictReader(stream))
    assert len(metrics) == 20 and int(metrics[-1]["steps"]) == 20
    assert all(math.isfinite(float(value)) for row in metrics for value in row.values() if value)
    saved_stats = load_file(str(checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"))
    for key in ("observation.state", "action"):
        for stat in ("mean", "std", "count"):
            np.testing.assert_allclose(saved_stats[f"{key}.{stat}"], stats[key][stat], atol=1e-6)
    policy = POLICY_CLASSES[cfg.policy.type].from_pretrained(checkpoint, config=cfg.policy)
    pre, post = make_pre_post_processors(cfg.policy, pretrained_path=str(checkpoint))
    dataset = LeRobotDataset("QYyyyyyyy/formal1_C50_nomaster", root=DATA,
                             delta_timestamps={"action": [i / 30 for i in range(100)]}, video_backend="pyav")
    for index, ep, frame in [(0, 0, 0), (7 * 718, 7, 0), (35900 - 1, 49, 717)]:
        example = dataset[index]
        assert int(example["episode_index"]) == ep and int(example["frame_index"]) == frame
        assert example["observation.state"].shape == (8,)
        assert all(example[camera].shape == (3, 480, 640) for camera in cfg.policy.image_features)
    batch = pre(next(iter(DataLoader(dataset, batch_size=8, collate_fn=collate))))
    policy.eval()
    with torch.inference_mode():
        predicted = policy.predict_action_chunk(batch)
        assert predicted.shape == (8, 100, 6) and torch.isfinite(predicted).all()
        assert torch.isfinite(post(predicted[:, 0])).all()
    del predicted
    optimizer = cfg.optimizer.build(policy.get_optim_params())
    policy.train()
    torch.cuda.synchronize()
    barrier.wait(timeout=120)
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        loss, losses = policy(batch)
        assert torch.isfinite(loss) and all(math.isfinite(float(value)) for value in losses.values())
        if model != "ACT":
            assert any(key.startswith("det_") for key in losses), losses
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10)
        assert torch.isfinite(gradient) and gradient > 0
        optimizer.step()
    torch.cuda.synchronize()
    result = {"model": model, "smoke_steps": 20, "reload_inference": "passed", "concurrent_updates": 10,
              "loss": float(loss), "loss_components": losses,
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
    (ROOT / f"preparation/{model}_resource_check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    preparation = ROOT / "preparation"
    assert not (preparation / "preparation_ready.json").exists()
    assert all((ROOT / f"C50_NOMASTER_{model}_smoke_s1000/train.done").read_text().strip() == "exit=0" for model in MODELS)
    context = mp.get_context("spawn")
    barrier = context.Barrier(3)
    workers = [context.Process(target=worker, args=(model, barrier)) for model in MODELS]
    for process in workers:
        process.start()
    samples, started = [], time.monotonic()
    while any(process.is_alive() for process in workers):
        if time.monotonic() - started > 300:
            for process in workers:
                if process.is_alive():
                    process.terminate()
            raise TimeoutError("Resource check timed out; full training not authorized by preparation gate")
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], text=True)
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True)
        used, total = map(int, gpu.strip().split(","))
        samples.append({"elapsed_s": round(time.monotonic() - started, 2), "used_mib": used,
                        "total_mib": total, "compute_processes": len(apps.strip().splitlines())})
        time.sleep(0.5)
    for process in workers:
        process.join()
    assert all(process.exitcode == 0 for process in workers), [process.exitcode for process in workers]
    assert max(sample["compute_processes"] for sample in samples) >= 3
    peak = max(sample["used_mib"] for sample in samples)
    assert total - peak >= 2048, f"Insufficient memory headroom: {total - peak} MiB"
    checks = [json.loads((preparation / f"{model}_resource_check.json").read_text()) for model in MODELS]
    ready = {"status": "passed", "completed_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
             "checks": checks, "concurrent_gpu_peak_mib": peak, "gpu_headroom_mib": total - peak,
             "training_script_sha256": sha(Path("scripts/train_c50_nomaster.sh")),
             "manifest_sha256": sha(DATA / "meta/nomaster_manifest.json"),
             "statistics_sha256": sha(DATA / "meta/stats.json"), "samples": samples}
    (preparation / "preparation_ready.json").write_text(json.dumps(ready, indent=2))
    print(f"PREPARATION_READY: peak={peak} MiB, headroom={total-peak} MiB, three workers passed", flush=True)
