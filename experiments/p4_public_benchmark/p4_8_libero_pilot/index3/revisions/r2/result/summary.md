# P4.8-r2 result

**结论：G2_PROTOTYPE_PROVENANCE_BLOCKED。停止条件已查明并有可复核证据；三个正式训练均未启动。**

原 P1 bank 依赖四个人工选定的杯内中心和35px椭圆采样。原脚本、存储provenance、bank SHA和Train90成员核验互相一致，不能在禁止新增目标ROI的LIBERO任务中无损复用。详见 prototype_lineage_audit.md。

完成：保留r0/G0/G1环境与仿真证据；r1/index2未运行；本轮按seed1一次冻结Train35（5476帧）/Dev9（1391帧），覆盖44集6867帧；服务器真实Train episode 813帧0完成单/双归一化**参考算式**测试；登记ACT/Legacy/Fixed三个阻断作业；形成最小兼容性proposal。

未完成且未伪造：模型实际输入hook、训练smoke、10k训练、checkpoint、吞吐测试、Dev9模型指标、闭环rollout、object on/off诊断。六个指标CSV只有表头代表NOT_RUN，没有可用于排名的数据。effective_config_*.json明确runtime=null，仅列合同目标而非生效配置。

服务器快照：2026-10-30T22:27:37+08:00；RTX5090，32607MiB，总显存占用2MiB，GPU0%；数据盘可用29,950,787,584 bytes（约27.9GiB）；未发现P4.5/P4.7/P4.8活动任务。无后台训练队列需要等待，亦无待续传模型。

交付：result/、decision.md、审计脚本、SHA清单与ZIP。STATUS及工作日志已同步。本轮没有提交/推送GitHub、上传HF、读取Final15或操作机械臂。下一步由用户与规划助手审批blocking_proposal.md中的原型来源兼容性，不自动越过G2。独立Builder/Red-Team均PENDING。

验证：审计脚本编译、JSON解析、split完整性及本轮STATUS/工作日志的定向diff检查通过。全工作树git diff --check因预先存在的无关指南尾随空格未通过，原输出保存于full_worktree_diff_check.log，未改动该指南。
