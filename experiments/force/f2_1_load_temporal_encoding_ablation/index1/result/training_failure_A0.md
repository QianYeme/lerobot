# 清理时发现的A0训练故障

> 当前：原失败仍有效且未确定根因。有界诊断15000步未复现；用户要求恢复并行后，启动一次同seed/同参数从0重试，screen F21_A0_RETRY1，输出results_A0_retry1/A0。原失败日志/目录不覆盖，诊断权重不用于正式结果；重试若再失败自动保存实际batch/权重/RNG后停止，不循环重启。

2026-10-19服务器19:05左右，A0退出码1，最后每100步日志为14200；实际异常可能在后续100步内，不能把14200当精确失败步。

异常发生在loss.backward之后的clip_grad_norm_(..., error_if_nonfinite=True)：total gradient norm non-finite。只能确认出现非有限梯度，尚未定位其根因；不是磁盘不足报错，也不能据此认定A0方法本身无效。证据保留在远程A0.log/A0.exit，另复制本地outputs/server_cleanup_20261019/A0_failure.log。

本次任务仅清理空间，未关闭检查、跳过坏batch、换seed、调学习率、重启或增减训练预算。A1/A2清理后仍在运行，快照21600/21700步。当前实验PARTIAL_TRAINING_FAILURE、F2_1_COMPLETE=false、科学UNRESOLVED；后续先诊断，再决定是否需要新的初始化一致重跑或合同修订。

清理只涉及其它历史实验的中间optimizer和已作废P1_s1权重，未删当前F2.1文件。A0在清理前就已退出。
