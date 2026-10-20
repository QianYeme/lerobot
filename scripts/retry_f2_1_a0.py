"""One user-authorized A0 retry, unchanged training math/seed/budget, with fault capture."""
import json
import shutil
import sys
from pathlib import Path

import torch

from diagnose_f2_1_a0 import cpu, gradient_audit, stats
from run_f2_1 import Experiment, shared_hash

BASE=Path('/root/autodl-tmp/F2_1_r1_20261019')


class CaptureExperiment(Experiment):
    def policy(self, variant):
        self.live_policy=super().policy(variant)
        initial=shared_hash(self.live_policy)
        assert initial=='a7105e4b47051dc74e2afe5247a4a20c127f4147a159e82a0cc86b18e163baf5'
        (self.out/'initialization.json').write_text(json.dumps(dict(shared_hash=initial,seed=1,variant='A0',
            previous_attempt='results/A0 (failed)',diagnostic_used_as_model=False),indent=2))
        return self.live_policy

    def batch(self, raw, variant):
        batch=super().batch(raw,variant)
        if self.live_policy.training:
            self.last_batch=batch
            self.cpu_rng=torch.get_rng_state()
            self.cuda_rng=torch.cuda.get_rng_state()
        return batch


def main():
    diagnosis=json.loads((BASE/'diagnosis_A0_summary.json').read_text())
    assert diagnosis['status']=='NOT_REPRODUCED_WITHIN_FIXED_BUDGET', 'Review a reproduced failure before restarting'
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True
    out=BASE/'results_A0_retry1'
    out.mkdir(exist_ok=False)
    for name in ['smoke.json','historical_metric_reproduction.json']:
        shutil.copy2(BASE/'results'/name,out/name)
    experiment=CaptureExperiment(BASE/'inputs',Path('/root/autodl-tmp/lerobot/formal4_data/kind_merged_nomaster'),out)
    try:
        experiment.train('A0')
    except Exception as error:
        tb=sys.exc_info()[2]
        context={}
        while tb:
            if tb.tb_frame.f_code.co_name=='train':context=tb.tb_frame.f_locals
            tb=tb.tb_next
        if context and hasattr(experiment,'last_batch'):
            policy=context['policy']
            torch.save(dict(policy=cpu(policy.state_dict()),optimizer=cpu(context['optimizer'].state_dict()),
                batch=cpu(experiment.last_batch),cpu_rng=experiment.cpu_rng,cuda_rng=experiment.cuda_rng,
                step=context.get('step')),out/'failure_capture.pt')
            report=dict(error=str(error),step=context.get('step'),gradients=gradient_audit(policy),
                        input_stats={k:stats(v) for k,v in experiment.last_batch.items()})
            (out/'failure_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        raise


if __name__=='__main__':main()
