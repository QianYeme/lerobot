# Builder 自查（非独立审查）

四模型来源和正式 step 均已核验，原 checkpoint 保留；导出未更新参数。
L1_160K 经用户明确追加，独立补充块，不修改原 decision。
共享 ACT 源码未热修复 DINO；F3 仅在部署包装层恢复独立 fixed 代码的 raw Top 边界。
真实 Dev15 比较证实四模型包装层与各自原始参考预测一致，reset 结果一致。
指南保留手动 Home、可修改平滑与限幅 CLI，不新增 Stage0/shadow/STOP 锁。
物理硬件预检与真机动作待现场；不将离线通过写成真机通过。
