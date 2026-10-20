# 最终数据/选择边界审计

仅固定Train36训练及归一化，固定Dev12评估/选模；Final12只查split元数据，不读取样本/视频。F1RM Formal5没有混入。load/state/action统计Train-only，state在模型与VAE均真实6D，current/master排除；A0无反馈token，A1当前load，A2因果W15首帧复制填充，不跨episode。

原8D P_hist只用于历史完整流水线复现，不warmstart新三组、不替代A0。各100k、同seed1/初始化公共hash与超参，在同20/40/60/80/100k上按冻结h0夹爪主指标选择；本次selected正确性和逐episode加权已复算。不用训练loss、100k末点或其它horizon后验换主指标。

同一Dev既选模又报告，所以不是独立泛化确认。A0原失败与同配置诊断不作模型选择候选，用户授权一次从0重试，重试与seed/稳定性风险披露；不隐藏额外计算，不宣称故障根因已修复。eval模式使用zero latent，合法未来action仅作监督目标，不进推理latent；padding排除。M1/M2是同观测离线扰动，不作为接触标签或因果力证据。

没有新采、机器人/部署参数修改、外部发布或push；5090成本不冒充4060/端到端表现。F1RM残余混杂未决、历史P1的输入语义和旧报告均不更改。
