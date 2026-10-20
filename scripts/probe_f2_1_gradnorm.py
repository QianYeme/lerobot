"""A deterministic discriminator, NOT a reproduction/diagnosis of the original A0 run."""
import json
from pathlib import Path

import torch

parameter=torch.nn.Parameter(torch.zeros(2,device='cuda'))
parameter.grad=torch.tensor([1e20,1e20],device='cuda')
report=dict(scope='SYNTHETIC_OPERATOR_PROBE_NOT_ORIGINAL_A0',gradient_elements_finite=bool(torch.isfinite(parameter.grad).all()),
            fp64_norm=float(torch.linalg.vector_norm(parameter.grad.double())))
try:
    value=torch.nn.utils.clip_grad_norm_([parameter],10.,error_if_nonfinite=True)
    report['fp32_clip']='FINITE';report['fp32_norm']=float(value)
except RuntimeError as error:
    report['fp32_clip']='RAISED_NONFINITE_NORM';report['error']=str(error)
out=Path('/root/autodl-tmp/F2_1_r1_20261019/gradnorm_operator_probe.json')
out.write_text(json.dumps(report,indent=2))
print(json.dumps(report))
