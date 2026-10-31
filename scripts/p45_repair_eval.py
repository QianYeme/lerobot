"""Evaluation-only recovery; never performs optimizer updates or robot I/O."""
import argparse
import json
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.run.startswith('VACT'):
        import p45_vanilla_worker as w
        from lerobot.policies.act.processor_act import make_act_pre_post_processors
        folder = w.OUT/'runs'/args.run
        checkpoint = folder/'checkpoints/080000/resume.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        policy = w.ACTPolicy(saved['config']).cuda()
        policy.load_state_dict(saved['model'])
        stats = w.statistics()
        pre, _ = make_act_pre_post_processors(saved['config'], stats)
        w.evaluate(policy, pre, stats, args.run, 80000)
    else:
        import p45_newgl_worker as w
        folder = w.OUT/'runs'/args.run
        checkpoint = Path(json.loads((folder/'latest.json').read_text())['checkpoint'])
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert saved['step'] == 20000
        assert saved['split_sha256'] == w.common.sha(w.PRIOR/'split_freeze_manifest.json')
        assert all(k in saved for k in ['model', 'optimizer', 'rng', 'epoch', 'cursor'])
        policy, cfg = w.make(args.run.startswith('NG-F'))
        policy.load_state_dict(saved['model'])
        opt = torch.optim.AdamW(policy.get_optim_params(), lr=1e-5, weight_decay=1e-4)
        opt.load_state_dict(saved['optimizer'])
        w.common.restore(saved['rng'])
        pre, _, stats = w.processors(cfg)
        score = w.evaluate(policy, pre, stats, args.run, saved['step'])
        w.common.append(folder/'selection.csv', dict(step=20000, primary=score, checkpoint=str(checkpoint)))
    report = dict(status='PASS', run=args.run, step=saved['step'],
                  checkpoint=str(checkpoint), sha256=w.common.sha(checkpoint) if hasattr(w, 'common') else w.sha(checkpoint),
                  optimizer_updates=0, final15_accessed=False, completed=time.time())
    (folder/'evaluation_repair.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
