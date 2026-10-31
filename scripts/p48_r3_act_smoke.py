"""Disposable native ACT real-Train100-update/recovery engineering gate, not pilot training."""
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path('/root/autodl-tmp/lerobot-main')
OLD = ROOT/'experiments/p4_public_benchmark/p4_8_libero_pilot/index1/result'
OUT = ROOT/'experiments/p4_public_benchmark/p4_8_libero_pilot/index3/result'
DATA = Path('/root/autodl-tmp/p48_data/lerobot_libero')
sys.path.insert(0, str(OLD/'workspace/lerobot/src'))


def main():
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import default_collate
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    assert '/index1/result/workspace/lerobot/src/' in inspect.getfile(ACTPolicy)
    torch.set_num_threads(4);torch.manual_seed(1);np.random.seed(1);random.seed(1)
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False;torch.use_deterministic_algorithms(True)
    split = json.loads((OUT/'dataset_split_manifest.json').read_text())
    rows = []
    for p in sorted((DATA/'data').rglob('*.parquet')):
        df = pd.read_parquet(p);df = df[df.episode_index.isin(split['train_episode_ids'])]
        if len(df):rows.append(df)
    frame = pd.concat(rows);assert len(frame) == 5476 and set(frame.task_index) == {24}
    stats = {}
    for key in ['observation.state', 'action']:
        values = np.stack(frame[key]);assert np.isfinite(values).all()
        stats[key] = dict(mean=torch.tensor(values.mean(0)), std=torch.tensor(values.std(0).clip(1e-6)))
    for key in ['observation.images.image','observation.images.image2']:
        stats[key] = dict(mean=torch.tensor([.485,.456,.406]).view(3,1,1), std=torch.tensor([.229,.224,.225]).view(3,1,1))
    torch.save(stats, OUT/'train35_statistics.pt')
    cfg = ACTConfig(input_features={'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(8,)),
                    **{key:PolicyFeature(type=FeatureType.VISUAL,shape=(3,256,256)) for key in ['observation.images.image','observation.images.image2']}},
                    output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(7,))})
    assert not hasattr(cfg,'object_conditioning_enabled') and not hasattr(cfg,'gripper_loss_weight')
    folder = OUT/'act_engineering_smoke';folder.mkdir(exist_ok=True)
    cfg.save_pretrained(folder/'effective_config')
    ACTConfig().save_pretrained(folder/'native_default_config')
    dataset = LeRobotDataset('lerobot/libero',root=DATA,episodes=[813],revision=split['revision'],
                             delta_timestamps={'action':[i/10 for i in range(cfg.chunk_size)]},video_backend='torchcodec')
    assert 813 in split['train_episode_ids']
    items = []
    for i in [0,20,40,60,80,100,120,140]:
        r = dataset[i];r = {k:r[k] for k in ['observation.state','action','action_is_pad','observation.images.image','observation.images.image2']}
        for key in ['observation.images.image','observation.images.image2']:r[key] = r[key].flip((-2,-1))
        items.append(r)
    raw = default_collate(items)
    pre, _ = make_act_pre_post_processors(cfg, stats)
    batch = pre(raw);assert batch['action'].shape == (8,100,7)
    policy = ACTPolicy(cfg).cuda();preset = cfg.get_optimizer_preset()
    opt = preset.build(policy.get_optim_params());clip = preset.grad_clip_norm
    def update(model, optimizer):
        model.train();optimizer.zero_grad(set_to_none=True);loss,_ = model(batch)
        assert torch.isfinite(loss);loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True)
        optimizer.step();return float(loss.detach()),float(grad)
    def rng():return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),python=random.getstate(),numpy=np.random.get_state())
    def restore(r):
        torch.set_rng_state(r['cpu']);torch.cuda.set_rng_state_all(r['cuda']);random.setstate(r['python']);np.random.set_state(r['numpy'])
    torch.cuda.synchronize();started = time.time()
    with (folder/'metrics.csv').open('w') as f:
        w = csv.DictWriter(f,fieldnames=['step','loss','gradient_norm','elapsed_s']);w.writeheader()
        for step in range(1,101):
            loss,grad = update(policy,opt)
            w.writerow(dict(step=step,loss=loss,gradient_norm=grad,elapsed_s=time.time()-started));f.flush()
            if step % 20 == 0:print('DISPOSABLE_ACT_SMOKE',step,loss,flush=True)
    checkpoint = folder/'resume_000100.pt'
    torch.save(dict(model=policy.state_dict(),optimizer=opt.state_dict(),rng=rng(),step=100,
                    config=json.loads((folder/'effective_config/config.json').read_text()),episode=813,indices=[0,20,40,60,80,100,120,140],
                    split_sha256=hashlib.sha256((OUT/'dataset_split_manifest.json').read_bytes()).hexdigest()),checkpoint)
    first = update(policy,opt);expected = {k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
    del policy,opt;torch.cuda.empty_cache()
    policy = ACTPolicy(cfg).cuda();opt = preset.build(policy.get_optim_params())
    saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
    policy.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer']);restore(saved['rng'])
    second = update(policy,opt)
    difference = max(float((expected[k]-v.detach().cpu()).abs().max()) for k,v in policy.state_dict().items())
    assert difference <= 1e-6,(difference,first,second)
    policy.eval()
    with torch.inference_mode():chunk = policy.predict_action_chunk(batch)
    assert chunk.shape == (8,100,7) and torch.isfinite(chunk).all()
    result = dict(status='NATIVE_ACT_ENGINEERING_SMOKE_PASS',updates=100,discarded_updates=True,
                  formal_pilot_updates=0,batch_size=8,episode=813,train_stats_frames=len(frame),
                  next_update_max_weight_difference=difference,paired_losses=[first[0],second[0]],
                  checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  parameters=sum(p.numel() for p in policy.parameters()),native_model_source=inspect.getfile(ACTPolicy),
                  normalization='Train35 state/action statistics; ImageNet RGB once; flipH/W once',
                  elapsed_s=time.time()-started,peak_memory_bytes=torch.cuda.max_memory_allocated(),
                  does_not_validate_generalization_or_closed_loop=True)
    (folder/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
