"""Read-only selected B0/B1 Dev12 evaluation, counterfactuals and inference cost."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from run_f2_1 import save
from run_f2_2 import GatedExperiment
from lerobot.policies.act.modeling_act import ACTPolicy


def measure(policy, batch):
    with torch.inference_mode():
        for _ in range(10):
            policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        warmup_peak = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        wall, gpu = [], []
        for _ in range(50):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start = time.perf_counter()
            begin.record()
            output = policy.predict_action_chunk(batch)
            end.record()
            torch.cuda.synchronize()
            wall.append(1000 * (time.perf_counter() - start))
            gpu.append(begin.elapsed_time(end))
        assert output.shape == (1, 100, 6) and torch.isfinite(output).all()
    return dict(parameters=sum(p.numel() for p in policy.parameters()),
        wall_ms_median=float(np.median(wall)), wall_ms_p95=float(np.percentile(wall, 95)),
        gpu_ms_median=float(np.median(gpu)), steady_peak_bytes=torch.cuda.max_memory_allocated(),
        warmup_peak_bytes=warmup_peak, batch_size=1, warmup=10, repeats=50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    out = base / 'delivery'
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    exp = GatedExperiment(base/'inputs', Path('/root/autodl-tmp/lerobot/formal4_data/kind_merged_nomaster'), base/'results')
    assert len(exp.spec['train']) == 36 and len(exp.spec['dev']) == 12
    raw_one = next(iter(exp.loader('dev', workers=0)))
    raw_one = {k:v[:1] for k,v in raw_one.items()}
    report, costs = {}, {}
    for variant in ['B0', 'B1']:
        directory = base/'results'/variant
        selection = json.loads((directory/'selected.json').read_text())
        checkpoint = Path(selection['checkpoint']).resolve()
        assert checkpoint.is_relative_to(directory.resolve())
        assert (checkpoint/'f2_1_normalization.json').read_bytes() == (base/'inputs/normalization.json').read_bytes()
        policy = ACTPolicy.from_pretrained(checkpoint).cuda().eval()
        assert policy.config.load_history_steps == 1
        assert policy.config.load_feature_gating == (variant == 'B1')
        assert tuple(policy.config.robot_state_feature.shape) == (6,)
        batch = exp.batch(raw_one, variant)
        batch.pop('action'); batch.pop('action_is_pad', None)
        torch.cuda.reset_peak_memory_stats()
        costs[variant] = measure(policy, batch)
        costs[variant].update(step=selection['step'], checkpoint=str(checkpoint), strict_reload=True,
            model_sha256=hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest())
        arrays = {k:[] for k in ['episode', 'frame', 'position', 'command', 'gate', 'prediction', 'target',
                                 'feature_zero_prediction', 'input_zero_prediction']}
        if variant == 'B1':
            arrays['gate_on_prediction'] = []
        with torch.inference_mode():
            for raw in exp.loader('dev'):
                batch = exp.batch(raw, variant)
                target = batch.pop('action')[:, 0]
                original_load = batch['observation.load']
                prediction = policy.predict_action_chunk(batch)[:, 0]
                if variant == 'B1':
                    normal_gate = batch['observation.load_gate']
                    batch['observation.load_gate'] = torch.zeros_like(normal_gate)
                    feature_zero = policy.predict_action_chunk(batch)[:, 0]
                    batch['observation.load_gate'] = torch.ones_like(normal_gate)
                    gate_on = policy.predict_action_chunk(batch)[:, 0]
                    arrays['gate_on_prediction'].append(gate_on.cpu().numpy())
                    batch['observation.load_gate'] = normal_gate
                else:
                    # M3 zero AFTER the biased MLP, retaining load token/position.
                    hook = policy.model.load_encoder.register_forward_hook(lambda module, inputs, output: torch.zeros_like(output))
                    try:
                        feature_zero = policy.predict_action_chunk(batch)[:, 0]
                    finally:
                        hook.remove()
                batch['observation.load'] = torch.zeros_like(original_load)
                input_zero = policy.predict_action_chunk(batch)[:, 0]
                batch['observation.load'] = original_load
                ids = raw['episode_index'].reshape(-1).tolist()
                frames = raw['frame_index'].reshape(-1).tolist()
                position = np.asarray([exp.cache[e]['state'][f, 5] for e,f in zip(ids, frames)])
                arrays['episode'].append(np.asarray(ids))
                arrays['frame'].append(np.asarray(frames))
                arrays['position'].append(position)
                arrays['command'].append(raw['action'][:, 0, 5].numpy())
                arrays['gate'].append((position <= exp.gate_spec['threshold']).astype(np.float32))
                for key, tensor in [('prediction', prediction), ('target', target),
                                    ('feature_zero_prediction', feature_zero), ('input_zero_prediction', input_zero)]:
                    arrays[key].append(tensor.cpu().numpy())
        arrays = {k:np.concatenate(v) for k,v in arrays.items()}
        assert len(arrays['episode']) == 5748
        assert set(arrays['episode']) == set(exp.spec['dev'])
        assert len(set(zip(arrays['episode'], arrays['frame']))) == 5748
        assert all(np.isfinite(v).all() for v in arrays.values())
        error = np.abs(arrays['prediction'] - arrays['target'])
        std = np.asarray(exp.norm['action']['std'])
        physical = error * std
        metrics = dict(primary=float(error[:,5].mean()), physical_gripper_mae=float(physical[:,5].mean()),
                       physical_arm_mae=float(physical[:,:5].mean()), normalized_all_l1=float(error.mean()), frames=5748)
        metrics['per_episode'] = {}
        for ep in exp.spec['dev']:
            mask = arrays['episode'] == ep
            metrics['per_episode'][str(ep)] = dict(primary=float(error[mask,5].mean()),
                physical_gripper_mae=float(physical[mask,5].mean()), physical_arm_mae=float(physical[mask,:5].mean()),
                frames=int(mask.sum()))
        stored = json.loads((directory/'evaluation'/f'{selection["step"]:06d}.json').read_text())
        assert abs(metrics['primary'] - stored['primary']) < 1e-5, 'Selected metric reproduction failed'
        np.savez_compressed(out/f'{variant}_selected_dev_predictions.npz', **arrays)
        counterfactuals = {}
        for name in ['feature_zero', 'input_zero'] + (['gate_on'] if variant == 'B1' else []):
            prediction_cf = arrays[name+'_prediction']
            delta = np.abs(prediction_cf - arrays['prediction'])
            counterfactuals[name] = dict(normalized_gripper_delta=float(delta[:,5].mean()),
                normalized_arm_delta=float(delta[:,:5].mean()), physical_gripper_delta=float((delta[:,5]*std[5]).mean()),
                physical_arm_delta=float((delta[:,:5]*std[:5]).mean()),
                counterfactual_primary=float(np.abs(prediction_cf[:,5]-arrays['target'][:,5]).mean()), frames=5748)
        if variant == 'B1':
            counterfactuals['gate_off'] = dict(counterfactuals['feature_zero'], alias='identical intervention to encoded feature zero')
            counterfactuals['gate_off']['inactive_region_max_abs_delta'] = float(np.abs(
                arrays['feature_zero_prediction'][arrays['gate']==0]-arrays['prediction'][arrays['gate']==0]).max())
            counterfactuals['gate_on']['active_region_max_abs_delta'] = float(np.abs(
                arrays['gate_on_prediction'][arrays['gate']==1]-arrays['prediction'][arrays['gate']==1]).max())
            assert counterfactuals['gate_off']['inactive_region_max_abs_delta'] < 1e-5
            assert counterfactuals['gate_on']['active_region_max_abs_delta'] < 1e-5
        report[variant] = dict(selection=selection, metrics=metrics, counterfactuals=counterfactuals,
            selected_metric_reproduction_abs_diff=abs(metrics['primary'] - stored['primary']))
        save(out/f'{variant}_selected_audit.json', report[variant])
        del policy, batch, prediction, target, feature_zero, input_zero
        if variant == 'B1':
            del gate_on
        gc.collect(); torch.cuda.empty_cache()
    save(out/'selected_evaluation.json', report)
    save(out/'latency_and_params.json', dict(gpu=torch.cuda.get_device_name(), variants=costs,
        scope='pure policy batch1; excludes camera/decode/preprocessing; RTX4060 unmeasured; warmup peak includes autotune'))


if __name__ == '__main__':
    main()
