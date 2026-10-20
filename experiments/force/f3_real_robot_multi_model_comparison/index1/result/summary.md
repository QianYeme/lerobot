# F3 r1 — FOUR_MODEL_TRANSFER_READY / REAL_ROBOT_NOT_RUN

按用户“刚刚的操作”范围：只发布模型、文件、指南并更新GitHub，不执行真机。
F3_R1_COMPLETE=false，科学比较尚无结果。

私有HF仓库：https://huggingface.co/QYyyyyyyy/F3_R1_MULTI_MODEL_SO101

固定revision：`e903b0a1489815077f7a0ef881bec4ca007b6b18`。

| 模型 | 确切来源 | 路径 |
|---|---|---|
| M0 | 历史8D P1_ocfix2_s1 100k | models/M0 |
| M1 | 6D A0 retry1 80k，无load/current | global |
| M2 | A1 60k，全局单帧load | models/M2 |
| M3 | 同M1全局模型＋C1-load10k局部MLP | global＋local/010000.pt |

M0权重与配置逐字节对照原HF固定revision，未猜测或替换P1。
四组身份及输入顺序见model_identity.json；实际发布信息见publication.json。
三项global及local checkpoint的远端hash均验证通过，manifest固定revision一致。

离线服务器CUDA：M0/M1/M2 strict-load、输入维度及有限synthetic forward通过；
M3同M1全局权重、disabled/shadow/arm不变/限幅与局部输出对照通过。
这些不是实机硬件Stage0，不能替代真实闭环安全审核。

所有现场共同配置仍PENDING，现有control_act_det.py未提供统一四组模型的
接线/完整遥测路径；新系统先完成这些再做真机。计划M3 shadow3次，之后
三个固定轮转M0→M1→M2→M3共12次正式试次，均未执行，不伪造trial结果。

旧F3-r0原目录/HF仓库不动，合同副本保存在../decision_r0.md。
不含A2/W15、B1门控、C1-no-load、FiLM或其他未授权模型。
未读Final12、未训练/微调或改变残差上限、未操作机械臂。
