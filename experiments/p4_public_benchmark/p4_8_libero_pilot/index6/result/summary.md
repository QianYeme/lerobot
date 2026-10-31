# P4.8-r6 result

**主结论：GATE_SUPPRESSION_SUPPORTED。** 正常门控把非零对象残差压到很小；同时存在初始梯度限制和原型持握/遮挡语义风险，尚未证明历史训练根因或增大gate的任务收益。

完成D0–D5：核验r5六个真实5k/10k checkpoint；锁定Dev27+Train9+原型参考4共40帧；两P1分别在5k/10k做全chunk复现/物理绕过、alpha剂量、无更新反向、初始化hash复现、热图/局部权重和输入/bank反事实；复核原30段视频与轨迹。没有optimizer更新、权重文件写入、新rollout、Final或r7。

关键结果：

- D0旧h0 ON/OFF复现误差0；gate0与物理绕过全chunk差0，重复fp32差0。当前训练模式本就是fp32，未发现测量假象。
- 10k Dev27中位raw residual L2约53.29/46.14，query约234.05/238.33；乘门控后残差/query约1.56e-6/8.06e-8。正常h0 arm raw差7.46e-8/1.90e-8，临时alpha=.01变为1.02e-4/7.72e-5。只说明通路能影响输出，未测试任务收益。
- 初始模型hash与r5一致；alpha=0使adapter/attention/slot梯度恰为0，gate/CNN梯度非零。10k末强制.01可显著提高对象内部梯度。内部权重和Adam状态确有变化，不能说完全没训练。
- Fixed在原型参考图ROI的权重集中明显高于Legacy，但这是参考图自一致性。不同阶段热图存在未持续跟随被拿起目标的疑点，不能直接推出Fixed能可靠定位。
- 只换对象Top/随机bank在正常gate下影响极小；换整个Top而保持Wrist/State时影响大。原型槽位排列不变是数学对称性。

所有数值、源图SHA、源权重路径与只读校验已交付。并行诊断短测约为顺序1.95倍；正式训练/评估资源未被干扰。原r5三组4/10记录保持原样。

下一步仅供审批：单变量非零gate初始化候选，保留所有其它控制，并先考虑热图语义风险；不自动执行。详细见mechanism_findings.md和next_experiment_proposal.md。独立Builder/Red-Team均PENDING。
