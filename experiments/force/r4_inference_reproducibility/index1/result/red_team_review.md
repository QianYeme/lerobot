# 执行方反证复核，不替代独立外审

- 同进程0不保证跨进程一致，固定fixture通过也不保证全量一致；本轮明确复现原B失败并补齐两seed全量D。
- 唯一控制为benchmark关闭，支持runtime算法选择相关原因，但未记录kernel ID，不能宣称逐算子证明。
- 重复性仅对当前GPU/版本/批量与冻结checkpoint有证据，不应保证所有未来机器或真机ensemble。
- 重算后global seed差异并未消失；不能把seed幸运或数据/优化问题都归咎已修复的评估噪声。三seed std与单个paired effect比较仍只是描述规则。
- Full60消耗历史保留集；全部指标不是held-out，更不是真机。local误差下降并不证明闭环力控或机械臂位置修正。
- 数据采集/安全/QC/未见测试必须在下一合同冻结；本轮不因PASS绕过审阅自动采集或训练。
- 原始historical失败、checkpoint、raw chunk和日志均保留，没有按分数换模型、阈值或丢弃失败样本。

独立外部Red-Team Review：PENDING。
