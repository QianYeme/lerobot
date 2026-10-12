# T1.1 Explicit Spatial Path Health

诊断模式：只读、冻结 P2/100000 checkpoint；未重训、未修改模型、未使用 Final Test 或真机。

## 结论

主诊断：`SCALE_STARVATION`；次诊断：`['OPTIMIZATION_REJECTION']`。

- learned beta：`0.00318068`。
- state token norm：`24.487650`。
- raw spatial embedding norm：`8.552184`。
- effective residual norm：`0.027202`。
- effective spatial/state：`1.111e-03`。
- beta=1.0 时平均动作变化：`0.004146`；learned beta 时仅 `1.275e-07`。
- x/y/xy/zero counterfactual 均在 beta=1.0 下产生可测变化，说明表征与融合路径本身不是完全失效。
- 输入方差不是零：train/dev 的 x、y、confidence 均有非零 std。
- hook 已捕获 state token 融合；DINO 仍冻结；中间 checkpoint 的梯度轨迹未记录，不能声称有完整梯度历史。

## 分类解释

主要原因是 `SCALE_STARVATION`：训练后 beta 从 0.01 收缩到约 0.00318，使有效空间残差仅为 state token 的约 `1.111e-03`。在 beta=0.5/1.0 时路径才出现明显动作敏感性，因此不能归因于坐标输入完全无效或必然的实现错误。`OPTIMIZATION_REJECTION` 作为次要解释：当前动作损失倾向主动压低该弱门控路径。

## 证据与限制

- `parameter_trajectory.json`：仅有 100k checkpoint；10k–90k checkpoint 未生成，梯度分量历史为 `NOT_RECORDED`。
- `implementation_check.json`：记录了 forward hook 和公式路径；LayerNorm 的内部前后差异本轮未直接捕获。
- 本轮只是路径健康诊断，不构成模型性能 PASS，也不授权下一轮架构或真机实验。
