# 当前数据边界

## r1当前范围

Formal4 parquet以episode_index过滤只读取Train36/Dev12。load/state/action统计仅Train；Final只查split元数据不读样本。F1RM/其他数据不混入。state真实6D，current/master不缓存；W15仅因果episode内首帧填充，不用接触标签。A0不得自动消费共用缓存里的load字段。

仅复核旧G0 Dev缓存的主指标归约，未产新模型比较成绩。预算及checkpoint选择在训练前写variant_configs；P_hist只作历史，不迁移其权重。模型级泄漏测试尚未执行，不能把数据测试当模型级通过。

## 以下为r0历史范围

本次仅读正式split元数据、Formal4数据集info、旧训练/评估报告、ACT/processor源码和HF固定revision的模型JSON。没有打开Formal4原始样本parquet或Final12图像/视频，没有生成新标签，没有按任何新Dev成绩选结构。

没有新训练/离线预测、contact gate、current分支、局部残差、自动夹紧或机器人调用。没有修改F1/F1P/F1Q/F1RM历史结论；F1RM仍为UNRESOLVED_RESIDUAL_CONFOUND。

预检记录config和原8D schema中的current存在，是发现泄漏风险，不代表本轮已经开始将current用于训练。后续必须澄清并冻结所有输入路径，尤其VAE和state token，才可执行。
