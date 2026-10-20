# 当前数据边界

本次仅读正式split元数据、Formal4数据集info、旧训练/评估报告、ACT/processor源码和HF固定revision的模型JSON。没有打开Formal4原始样本parquet或Final12图像/视频，没有生成新标签，没有按任何新Dev成绩选结构。

没有新训练/离线预测、contact gate、current分支、局部残差、自动夹紧或机器人调用。没有修改F1/F1P/F1Q/F1RM历史结论；F1RM仍为UNRESOLVED_RESIDUAL_CONFOUND。

预检记录config和原8D schema中的current存在，是发现泄漏风险，不代表本轮已经开始将current用于训练。后续必须澄清并冻结所有输入路径，尤其VAE和state token，才可执行。
