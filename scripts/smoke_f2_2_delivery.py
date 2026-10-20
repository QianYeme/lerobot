"""Test postprocessing GPU paths on existing smoke checkpoints, never train."""
import json
from pathlib import Path

import torch

from evaluate_f2_2_selected import measure
from run_f2_2 import GatedExperiment
from lerobot.policies.act.modeling_act import ACTPolicy


def main():
    base=Path('/root/autodl-tmp/F2_2_r0_20261019')
    torch.set_num_threads(4)
    exp=GatedExperiment(base/'inputs',Path('/root/autodl-tmp/lerobot/formal4_data/kind_merged_nomaster'),base/'results')
    raw=next(iter(exp.loader('dev',workers=0)))
    raw={k:v[:1] for k,v in raw.items()}
    checks={}
    for variant in ['B0','B1']:
        policy=ACTPolicy.from_pretrained(base/'results/smoke_checkpoints'/variant).cuda().eval()
        batch=exp.batch(raw,variant); batch.pop('action')
        with torch.inference_mode():
            hook=policy.model.load_encoder.register_forward_hook(lambda m,i,o:torch.zeros_like(o))
            try:
                feature_zero=policy.predict_action_chunk(batch)
            finally:
                hook.remove()
            if variant=='B1':
                batch['observation.load_gate'].zero_()
                assert torch.equal(feature_zero,policy.predict_action_chunk(batch))
            else:
                encoded=policy.model.load_encoder(batch['observation.load'])
                assert torch.isfinite(encoded).all()
        assert tuple(policy.config.robot_state_feature.shape)==(6,)
        checks[variant]=measure(policy,batch)
        del policy,batch;torch.cuda.empty_cache()
    (base/'delivery_smoke.json').write_text(json.dumps(dict(status='PASS',checks=checks),indent=2))
    print('PASS: selected strict load / encoded-zero hook / benchmark paths; no training')


if __name__=='__main__':
    main()
