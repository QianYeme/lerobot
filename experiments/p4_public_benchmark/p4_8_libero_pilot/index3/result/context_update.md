## 2026-10-30 P4.8-r3自动原型实际执行：G2未通过，正式训练0步

- r3逐字归档index3/decision.md；原r2 decision/result完整迁至index3/revisions/r2，旧ZIP保留。r3明确使用自动Train-only方法，未以旧Codex看图坐标绕过禁止条款。
- 冻结Train35/Dev9不变；18个Train双相机共享视频分片SHA通过。GPU完成35集315帧事件窗口DINO提取/光流跟踪；23发现集获得12候选/7集，12个Train-only保留proxy集获得7候选/5集。原型全部来自Train，未用Dev图像/仿真物体GT。
- 生成4×384诊断bank，四源1171/1187/1151/850；完整重提取/选取重复SHA一致1d78709651259d73edc5f1d00a687f30ce0c251b3e79b5045d73478673953d66；背景/预运动邻域proxy间隔.02684/.05823通过。但12<实现前预注册16候选覆盖门槛；事后时序图亦提示机械臂/背景污染，故G2_AUTO_PROTOTYPE_UNRELIABLE，不降低阈值、不重选帧、不启动正式训练。16为执行者预注册启发式，非decision直接给定数值。
- 原生官方ACT真实Train100步工程smoke通过，完整optimizer/RNG恢复下一步权重差0；仅state8/action7/双256相机接口override，训练超参原生默认。所有更新丢弃，不计三组10k预算。实际FrozenDINO单/双前向数值probe通过，但非完整P1/CNN不变验证。
- 原生仿真20Hz单步实测.05s；数据编码timestamp10fps。没有重复/插值或改控制频率；采集/转换物理时基仍未证实，TIMEBASE_BLOCKED（原生接口工程PASS），没有伪称物理跨度相等。
- 三组正式训练均NOT_STARTED、无Dev9模型指标/策略rollout/成功率。nohup诊断队列完成退出；23:00服务器GPU2MiB/0%，磁盘约26.9GiB空闲，无活动训练。大缓存/工程checkpoint留服务器并有SHA指针，未删除证据。
- [r3交付摘要](../experiments/p4_public_benchmark/p4_8_libero_pilot/index3/result/summary.md)及blocking_proposal.md待用户与规划助手决定下一步；不建议仅降低12/16门槛就训练。独立Builder/Red-Team PENDING；未改P4.7、未Final15/真机/HF或Git发布。
