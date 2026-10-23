# R4运行记录（2026-10-23）

结束更新：后台计算与执行方打包均已结束，GPU空闲；两seed全量D差0，技术gate通过。最新见[result](result/summary.md)及[交付包](R4_REVIEWED_result_20261023.zip)。下文为原启动/阶段记录，不代表仍在运行；R4_COMPLETE仍false是等待独立外审，不是技术重复性失败。

合同为同目录decision_R4_INFERENCE_REPRODUCIBILITY_AND_FINAL_DATASET_GATE.md，r0，只读。服务器weste:20143；工作目录`/root/autodl-tmp/R4_20261023`；使用现有RTX5090，全部GPU推理在服务器执行。本地只实现脚本与CPU门槛/配置回归。

已完成：

- 历史0.117225647差异从原cache重现，最差episode3/frame77；原权重及输出不覆盖。
- historical同张量同进程A：三个固定批次各重复3次，完整chunk差0。
- historical跨进程B：相同输入hash，最早观测到不同的hook为model.backbone；完整chunk最大差0.327590942，h0最大差0.130935669。
- 单变量对照只把cudnn.benchmark True→False，matmul/cudnn TF32仍True，不固定或改变原训练seed；新A/B均为0。
- benchmark_off的C：重新加载同固定IDs，decoded images、归一化图像/state/load、输入及feature hooks一致，chunk差0。

当前后台D：seed3两次Full60完整回放，然后seed1 A/B/C/D验证，再fresh G36/S2/S4并重算F5.1关键指标。审计按层和按进程串行，避免GPU竞争干扰算法选择定位；不是训练任务。每个长阶段有日志，失败停止、保留现场；盘预留15GiB。旧F5.1源/模型/报告不改，新分析写入R4隔离view。

入口：`scripts/run_r4_repro.py`、`scripts/queue_r4_repro.py`、两个launch脚本。nohup父队列日志`queue.log`，状态`result/queue_state.json`；阶段日志在logs/，raw证据在raw/。不要未检查状态就重复启动。

本页只记录已核实的局部结果与后台计划。R4_COMPLETE=false；R4_REPRODUCIBILITY_PASS尚不能判true，须两seed全量D通过。暂停进入最终数据采集、SWA/SAM、多seed训练、真机或发布。下一轮数据方案只是预规划，不是本轮执行。
