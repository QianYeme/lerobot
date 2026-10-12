# T0 Position Discovery：B0 ACT vs 修复后 P1 OC-ACT

## 结论范围

本轮只使用 D1 固定 Dev split 的 12 个 episode；未使用 Final Test，也未进行真机测试。首轮 `P1_s1` 因远程未同步 OC-ACT 模块而排除，不进入科学结论。

## 训练与 Dev L1

两模型均完成 100,000 steps、seed=1、36 train episodes，退出码为 0，无 NaN/Inf/OOM。B0 的 Dev 最优为 40k，inference L1 为 0.241336；修复后 P1 的 Dev 最优为 100k，inference L1 为 0.230687，相对 B0 降低约 4.43%。

## 位置分层

在最优 checkpoint 上，P1 各视觉空间分层的第 0 步动作误差约为 0.060–0.101，B0 约为 0.103–0.170；P1 的分层误差整体更低且更均匀。详细数值见 `mechanism_fix/B0_040000.json` 与 `mechanism_fix/P1_100000.json`。

## OC-ACT 机制反事实

P1 学到的 gate 为 -0.02939。对同一批 Dev 输入：目标分支置零导致平均动作变化 0.07452，批内打乱导致 0.02036，空间交换（翻转 token 的 x 坐标）导致 0.00103。说明显式目标分支已实际影响动作，但当前模型对绝对 x 坐标的直接敏感性仍弱；不能把总 L1 改善直接解释为“已经充分使用空间坐标”。

## 判定

当前 T0 判定为 **PARTIAL**：OC-ACT 的真实注入和 Dev 改善得到验证，但空间交换效应较小，需要先确认位置诊断定义/坐标编码，再决定是否进入后续 Final Test。下一步不自动进入真机测试。
