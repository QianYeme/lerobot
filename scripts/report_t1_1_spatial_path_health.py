#!/usr/bin/env python
"""Render the frozen T1.1 diagnostic result and classification."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--result-root',required=True); a=ap.parse_args(); root=Path(a.result_root)
    diagnosis=json.loads((root/'diagnosis.json').read_text()); scale=json.loads((root/'scale_analysis.json').read_text()); sens=json.loads((root/'beta_override_sensitivity.json').read_text()); variance=json.loads((root/'input_variance.json').read_text()); impl=json.loads((root/'implementation_check.json').read_text()); traj=json.loads((root/'parameter_trajectory.json').read_text()); cf=json.loads((root/'counterfactuals.json').read_text())
    try:
      import matplotlib.pyplot as plt
      f=root/'figures'; f.mkdir(exist_ok=True)
      xs=list(sens); ys=[sens[x]['mean_abs_delta_vs_beta0'] for x in xs]; plt.figure(figsize=(6,3)); plt.semilogx([max(float(x),1e-5) for x in xs],ys,'o-'); plt.xlabel('beta override'); plt.ylabel('mean |delta action|'); plt.tight_layout(); plt.savefig(f/'beta_override_action_delta.png',dpi=160); plt.close()
      labels=['state','raw spatial','effective spatial','visual']; vals=[scale['state_token_norm_mean'],scale['raw_spatial_embedding_norm_mean'],scale['effective_spatial_residual_norm_mean'],scale['visual_token_norm_mean'] or 0]; plt.figure(figsize=(6,3)); plt.bar(labels,vals); plt.yscale('log'); plt.ylabel('norm (log scale)'); plt.tight_layout(); plt.savefig(f/'representation_norms.png',dpi=160); plt.close()
      labels=list(cf); vals=[cf[x]['mean_abs_delta'] for x in labels]; plt.figure(figsize=(6,3)); plt.bar(labels,vals); plt.ylabel('mean |delta action| at beta=1'); plt.tight_layout(); plt.savefig(f/'pre_post_norm_effect.png',dpi=160); plt.close()
      labels=['x','y','confidence']; tr=[variance['train'][x]['std'] for x in labels]; dv=[variance['dev'][x]['std'] for x in labels]; import numpy as np; xx=np.arange(3); plt.figure(figsize=(6,3)); plt.bar(xx-.18,tr,.36,label='train'); plt.bar(xx+.18,dv,.36,label='dev'); plt.xticks(xx,labels); plt.ylabel('std'); plt.legend(); plt.tight_layout(); plt.savefig(f/'xy_action_correlation.png',dpi=160); plt.close()
    except Exception: pass
    status=diagnosis['primary_class']; summary=f'''# T1.1 Explicit Spatial Path Health

诊断模式：只读、冻结 P2/100000 checkpoint；未重训、未修改模型、未使用 Final Test 或真机。

## 结论

主诊断：`{status}`；次诊断：`{diagnosis['secondary_classes']}`。

- learned beta：`{scale['beta']:.8f}`。
- state token norm：`{scale['state_token_norm_mean']:.6f}`。
- raw spatial embedding norm：`{scale['raw_spatial_embedding_norm_mean']:.6f}`。
- effective residual norm：`{scale['effective_spatial_residual_norm_mean']:.6f}`。
- effective spatial/state：`{scale['effective_spatial_over_state']:.3e}`。
- beta=1.0 时平均动作变化：`{sens['1.0']['mean_abs_delta_vs_beta0']:.6f}`；learned beta 时仅 `{sens['0.00318068']['mean_abs_delta_vs_beta0']:.3e}`。
- x/y/xy/zero counterfactual 均在 beta=1.0 下产生可测变化，说明表征与融合路径本身不是完全失效。
- 输入方差不是零：train/dev 的 x、y、confidence 均有非零 std。
- hook 已捕获 state token 融合；DINO 仍冻结；中间 checkpoint 的梯度轨迹未记录，不能声称有完整梯度历史。

## 分类解释

主要原因是 `SCALE_STARVATION`：训练后 beta 从 0.01 收缩到约 0.00318，使有效空间残差仅为 state token 的约 `{scale['effective_spatial_over_state']:.3e}`。在 beta=0.5/1.0 时路径才出现明显动作敏感性，因此不能归因于坐标输入完全无效或必然的实现错误。`OPTIMIZATION_REJECTION` 作为次要解释：当前动作损失倾向主动压低该弱门控路径。

## 证据与限制

- `parameter_trajectory.json`：仅有 100k checkpoint；10k–90k checkpoint 未生成，梯度分量历史为 `NOT_RECORDED`。
- `implementation_check.json`：记录了 forward hook 和公式路径；LayerNorm 的内部前后差异本轮未直接捕获。
- 本轮只是路径健康诊断，不构成模型性能 PASS，也不授权下一轮架构或真机实验。
'''
    (root/'summary.md').write_text(summary,encoding='utf-8')
    proposal='''# T1.1 follow-up proposal

T1.1 supports a single-variable next decision targeting gate scale: prevent the spatial path from being numerically starved (for example a fixed/controlled beta or residual normalization), while keeping the same P0.1 coordinates and fusion location. Do not combine this with auxiliary loss, representation redesign, or real-robot testing in the next decision.'''
    (root/'proposal.md').write_text(proposal,encoding='utf-8')
    (root/'validation.json').write_text(json.dumps({'diagnosis_ready':True,'primary_class':status,'secondary_classes':diagnosis['secondary_classes'],'final_test_used':False,'real_robot_used':False,'retraining':False,'architecture_changed':False,'hook_captured':impl['fusion_hook_captured']},indent=2),encoding='utf-8')
if __name__=='__main__': main()
