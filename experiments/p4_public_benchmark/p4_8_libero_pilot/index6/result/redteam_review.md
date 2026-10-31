# Executor Red-Team precheck

- 强制alpha、随机bank、混合Top/Wrist/State全部是离线诊断/OOD探针，不能当正式策略/成功率；当前r5仍三者4/10。
- Gate抑制的模型内直接效应被验证，不等于小gate的历史唯一根因；大gate改变动作不等于动作更好。
- 零初始内部梯度为0是代数和hash校验后的实测；Adam自适应且5k/10k参数发生变化，不能说内部完全没有学。
- 参考ROI权重只是在建库图上的自一致性；不是独立检测准确率。看过的Dev热图不能用于偷偷重建bank。ROI外不全是背景。
- Prototype permutation为max匹配不变性，不是有效性消融；随机patch的语义无关性未有GT保证。
- 主干Top替换敏感不证明正确目标定位，物理不一致输入需OOD标签；D5只能给取样可见过程及不确定性，不能把flag当失败阶段。
- r7只有候选、未实施；不以10个相同结果宣称模型等价。

**Independent Red-Team PENDING**。此为执行者风险自检，未虚构外部审查。
