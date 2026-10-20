"""Read-only selected-checkpoint loadability/latency audit; not a robot test."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from run_f2_1 import Experiment
from lerobot.policies.act.modeling_act import ACTPolicy

BASE=Path('/root/autodl-tmp/F2_1_r1_20261019')


def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True
    exp=Experiment(BASE/'inputs',Path('/root/autodl-tmp/lerobot/formal4_data/kind_merged_nomaster'),BASE/'results')
    raw=next(iter(exp.loader('dev',workers=0)))
    raw={k:v[:1] for k,v in raw.items()}
    directories=dict(A0=BASE/'results_A0_retry1/A0',A1=BASE/'results/A1',A2=BASE/'results/A2')
    result={}
    for name,directory in directories.items():
        selected=json.loads((directory/'selected.json').read_text())
        checkpoint=Path(selected['checkpoint'])
        assert (checkpoint/'f2_1_normalization.json').read_bytes()==(BASE/'inputs/normalization.json').read_bytes()
        policy=ACTPolicy.from_pretrained(checkpoint).to('cuda').eval()
        assert policy.config.robot_state_feature.shape[0]==6
        assert policy.config.load_history_steps==dict(A0=0,A1=1,A2=15)[name]
        batch=exp.batch(raw,name);batch.pop('action');batch.pop('action_is_pad',None)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for _ in range(10):policy.predict_action_chunk(batch)
            torch.cuda.synchronize()
            warmup_peak=torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            timings=[];gpu=[]
            for _ in range(50):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                started=time.perf_counter();begin.record()
                prediction=policy.predict_action_chunk(batch)
                end.record();torch.cuda.synchronize()
                timings.append(1000*(time.perf_counter()-started));gpu.append(begin.elapsed_time(end))
        assert prediction.shape==(1,100,6) and torch.isfinite(prediction).all()
        result[name]=dict(step=selected['step'],checkpoint=str(checkpoint),parameters=sum(p.numel() for p in policy.parameters()),
            model_sha256=hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest(),
            batch_size=1,warmup=10,repeats=50,cpu_wall_ms_median=float(np.median(timings)),
            cpu_wall_ms_p95=float(np.percentile(timings,95)),gpu_ms_median=float(np.median(gpu)),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),strict_load=True,finite_output=True)
        result[name]['warmup_peak_bytes']=warmup_peak
        result[name]['memory_scope']='steady peak after warmup; first-process cuDNN autotune may inflate warmup peak'
        del policy,batch,prediction;torch.cuda.empty_cache()
    report=dict(status='PASS',gpu=torch.cuda.get_device_name(),scope='policy inference only, excludes camera/video decode and normalization; RTX4060 unmeasured',variants=result)
    (BASE/'results/selected_benchmark.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':main()
