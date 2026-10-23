# R4阶段进度：运行时差异已在最小案例定位，全量验证运行中

2026-10-23。R4_COMPLETE=false。当前不是最终result，不宣称数据采集gate已打开。

原历史cache最大h0差0.117225647，最差ep3/frame77。同一已保存张量在同一进程重复前向差0；跨新进程复现差异，最早观测到的不同模块是model.backbone（这不是该模块内部逐算子定位）。historical跨进程最大完整chunk差0.327590942，h0差0.130935669。

只关闭cudnn.benchmark，保持TF32、权重、归一化与模型语义不变后，三个固定批次的A/B差均为0，C原始数据预处理复核也为0。这支持cuDNN benchmark跨进程算法选择相关的runtime原因，不等于已经记录到具体内核ID。还需要D全量回放和seed1验证才能给出正式REPRODUCIBILITY_FIXED_RUNTIME判定。

服务器后台正按顺序完成D及必要补证；详细入口见[RUNNING.md](../RUNNING.md)。后续若两seed通过，再重算冻结的F5.1关键指标，所有新结果只写R4，不覆盖旧报告。训练、实际部署参数和真机均不改变；120–150段最终数据采集等仍未执行。

完整权重/raw/日志留服务器`/root/autodl-tmp/R4_20261023`；本地阶段证据见`evidence/`，不把局部通过冒充R4验收完成。执行方Builder/Red-Team最终复核和外部审阅尚待完成。
