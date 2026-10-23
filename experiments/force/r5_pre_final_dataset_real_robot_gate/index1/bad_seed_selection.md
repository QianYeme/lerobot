# R5弱seed对照冻结（真机前）

M5固定为 **G60-S4/80k**。不测试S1/S2，也不根据真机表现重新选择。

依据R4修复后共同诊断：S4夹爪MAE=2.614576、arm MAE=3.310617，为四个Full60 global seed中两项误差均最高；边缘幅度偏差0.469104，比S3的0.164667更高；方向余弦0.805936，比S3的0.916230低，空间多条件标签UNRESOLVED。相比S1，S4的两个整体误差更差且边缘幅度偏差更大，符合R5“combined arm/gripper and spatial diagnostics”的预先选择规则。

S4的匹配动作距离并非最小，也不是所有空间指标都最差，故不能称各指标一致最差。这里只是固定一个明确较弱的综合离线对照，不是统计离群点或真机弱性能证明。

权重SHA256：1277d98f221a69df2dcfd04361c34828f27799cf4794c25c75a6a83a3a396c8d。
训练Full60、seed4、80k，无local；全部训练checkpoint保持原样。R4修复后的结果来源为仓库experiments/force/r4_inference_reproducibility/index1/result。本轮真机尚未运行。
