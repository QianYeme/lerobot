"""F2.2: reuse frozen F2.1 training/metrics, change only load-feature gating."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from run_f2_1 import Experiment, save
from lerobot.utils.constants import ACTION


def weight_hash(policy):
    digest = hashlib.sha256()
    for key, value in policy.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class GatedExperiment(Experiment):
    def __init__(self, root, data, out):
        super().__init__(root, data, out)
        self.gate_spec = json.loads((root / 'gate_config.json').read_text())

    def config(self, variant):
        cfg = super().config('A1')
        cfg.load_feature_gating = variant == 'B1'
        return cfg

    def batch(self, raw, variant):
        batch = super().batch(raw, 'A1')
        if variant == 'B1':
            ids = raw['episode_index'].reshape(-1).tolist()
            frames = raw['frame_index'].reshape(-1).tolist()
            pos = np.asarray([self.cache[e]['state'][f, 5] for e, f in zip(ids, frames)])
            batch['observation.load_gate'] = torch.tensor(
                (pos <= self.gate_spec['threshold']).astype(np.float32)[:, None], device='cuda')
        return batch

    @torch.no_grad()
    def evaluate(self, policy, variant, loader, mechanisms=False):
        # Parent metric/horizon reduction unchanged. It also reports normalized-input
        # zero and +/-0.1 load sensitivity; these are NOT encoded-feature zero.
        score = super().evaluate(policy, variant, loader, mechanisms=mechanisms)
        if mechanisms and variant == 'B1':
            activity, effects = {}, []
            for raw in loader:
                batch = self.batch(raw, variant)
                batch.pop(ACTION)
                pred = policy.predict_action_chunk(batch)[:, 0]
                normal = batch['observation.load_gate']
                for ep, active in zip(raw['episode_index'].reshape(-1).tolist(), normal[:, 0].tolist()):
                    sums = activity.setdefault(str(ep), [0, 0])
                    sums[0] += active
                    sums[1] += 1
                for name, forced in [('gate_off_encoded_zero', torch.zeros_like(normal)),
                                     ('gate_on', torch.ones_like(normal))]:
                    batch['observation.load_gate'] = forced
                    delta = (policy.predict_action_chunk(batch)[:, 0] - pred).abs()
                    effects.append(dict(mode=name, gripper=float(delta[:, 5].sum()),
                        arm=float(delta[:, :5].mean(-1).sum()), n=len(normal)))
                # M3 encoded-feature-zero is mathematically identical to M1.
                # Parent 'zero' separately tests normalized input at Train mean.
            score['gate_activity'] = {ep:dict(active=s[0], frames=s[1], fraction=s[0]/s[1])
                                      for ep, s in activity.items()}
            score['gate_counterfactuals'] = effects
            score['M3'] = 'encoded-feature zero equals M1; normalized-input zero separately in mechanisms'
        return score

    def smoke(self):
        raw = next(iter(self.loader('train', workers=0)))
        results, initial = {}, []
        for variant in ['B0', 'B1']:
            policy = self.policy(variant)
            initial.append(weight_hash(policy))
            batch = self.batch(raw, variant)
            policy.eval()
            inference = {k:v for k,v in batch.items() if k != ACTION}
            with torch.no_grad():
                reference = policy.predict_action_chunk(inference)
                if variant == 'B1':
                    inference['observation.load_gate'] = torch.ones((4, 1), device='cuda')
                    on = policy.predict_action_chunk(inference)
                    policy.config.load_feature_gating = False
                    always = policy.predict_action_chunk(inference)
                    assert torch.equal(on, always), 'Gate-on must exactly equal ungated encoder'
                    policy.config.load_feature_gating = True
                    inference['observation.load_gate'].zero_()
                    off = policy.predict_action_chunk(inference)
                    inference['observation.load'] = inference['observation.load'] + 10
                    assert torch.equal(off, policy.predict_action_chunk(inference)), 'Off leaked load'
            # Explicit encoder-gradient gate test, independent of sampled gate activity.
            if variant == 'B1':
                encoded = policy.model.load_encoder(batch['observation.load'])
                (encoded * 0).sum().backward()
                assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0
                           for p in policy.model.load_encoder.parameters())
                policy.zero_grad(set_to_none=True)
            policy.train()
            loss, _ = policy(batch)
            assert torch.isfinite(loss)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10, error_if_nonfinite=True)
            assert policy.model.load_encoder[0].weight.grad.abs().sum() > 0
            optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-5, weight_decay=1e-4)
            optimizer.step()
            assert weight_hash(policy) != initial[-1]
            checkpoint = self.out / 'smoke_checkpoints' / variant
            policy.save_pretrained(checkpoint)
            policy.eval()
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.act.modeling_act import ACTPolicy
            cfg = PreTrainedConfig.from_pretrained(checkpoint)
            reloaded = ACTPolicy.from_pretrained(checkpoint, config=cfg).cuda().eval()
            test = self.batch(raw, variant)
            test.pop(ACTION)
            with torch.no_grad():
                assert torch.equal(policy.predict_action_chunk(test), reloaded.predict_action_chunk(test))
            results[variant] = dict(loss=float(loss.detach()), grad_norm=float(grad),
                parameters=sum(p.numel() for p in policy.parameters()), initial_hash=initial[-1],
                strict_reload=True, shape=list(reference.shape))
            del policy, reloaded, optimizer, batch, loss
            gc.collect()
            torch.cuda.empty_cache()
        assert initial[0] == initial[1]
        save(self.out / 'smoke.json', dict(status='PASS', variants=results, full_initialization_equal=True))


def main():
    parser = argparse.ArgumentParser()
    for key in ['root', 'data', 'out']:
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--mode', choices=['smoke', 'train'], required=True)
    parser.add_argument('--variant', choices=['B0', 'B1'])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    experiment = GatedExperiment(args.root, args.data, args.out)
    if args.mode == 'smoke':
        experiment.smoke()
    else:
        experiment.train(args.variant)


if __name__ == '__main__':
    main()
