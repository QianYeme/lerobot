## 2026-10-30 P4.8-r2 原型来源审计：G2-B阻断交付

- 新合同逐字归档到 `experiments/p4_public_benchmark/p4_8_libero_pilot/index3/decision.md`，不覆盖index1；服务器无index2或活动benchmark训练。
- 查明原P1 bank来自Train90四帧人工指定中心+35px椭圆patch均值，bank SHA `787a7084f1a387c71caaae635618458ec620b750d258df5df52578b1e7bc272f`；源码/provenance/缓存一致，四集Train成员核验通过。不是无目标先验自动原型发现。
- 因r2禁止新增ROI且不能用旧杯bank，按第4节B与第6节1停止正式训练；ACT/LEGACY/FIXED三个10k作业登记BLOCKED，均0步，无伪造checkpoint/Dev/rollout。未擅自使用聚类、随机patch、sim GT或人工ROI。
- r2授权一次冻结seed1 Train35/Dev9（5476/1391帧）并校验无重叠；真实Train ep813/f0仅运行单/双归一化参考算式，非DINO/策略forward或有效配置验证。历史adapter=256、dim_model=512核对通过；10FPS数据/20Hz控制未修改，闭环时基仍需下一合同明确。
- 交付 [index3/result/summary.md](../experiments/p4_public_benchmark/p4_8_libero_pilot/index3/result/summary.md)、来源审计及最小兼容性proposal；独立Builder/Red-Team PENDING。下一步由用户与规划助手决定是否授权少量Train目标先验，未自动进入G3。
- 服务器22:27快照：RTX5090，GPU2MiB/0%，数据盘约27.9GiB空闲，无活动P4.5/P4.7/P4.8训练。未改P4.7，未读Final15，未真机，未Git/HF发布。
