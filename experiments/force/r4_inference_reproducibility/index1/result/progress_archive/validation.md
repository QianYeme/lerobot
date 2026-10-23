# R4阶段验证（运行中，非最终验收）

- 服务器实际GPU：historical A通过、historical B失败；单独关闭cudnn.benchmark后A/B/C通过。原始分组CSV和两组运行时配置已下载至evidence/。
- D0第一次Full60/28,739帧已完成，D1第二次正在后台执行；尚未比较，不能说D通过。
- seed1第二seed与修复后关键指标重算尚未完成，R4_REPRODUCIBILITY_PASS暂未成立，R4_COMPLETE=false。
- 本地CPU定向回归8项通过（test_r4_repro_contract.py与test_f5_1_analysis.py，隔离根硬件conftest）；测试覆盖严格<0.001门槛、完整chunk而非仅h0、非有限输出拒绝、benchmark单变量控制。不是CUDA或硬件集成证据。
- git diff --check通过；旧训练/模型/数据/科研门槛/实际部署参数未改，原F5报告不覆盖。
- 后台队列和交付follower已启动；失败保留日志并停止，不自动添加配置、训练或新科研分支。最终报告由后台生成后仍需检查，并等待独立Builder/Red-Team。
