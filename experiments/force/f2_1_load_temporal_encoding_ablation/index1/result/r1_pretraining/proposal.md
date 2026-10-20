# F2.1 输入语义修订提议（等待批准）

> r1已批准干净6D设计，下文r0提议不再等待同一批准。当前仅缺可用GPU服务器。取得新SSH后，在当前合同内实现模型、校验数据/prototype及共享初始化、做真实GPU smoke和完整指标复现，再并行screen训练；不绕过门或warmstart P_hist。旧r0原件在r0_preflight/。

## 推荐：做干净的三个对照

建议在新revision明确：P1指视觉/object backbone与动作建模结构，而不是原8D checkpoint原样推理。

- A0：5个臂关节位置+gripper.pos，不含load/current。
- A1：同一位置状态+单帧load分支。
- A2：同一位置状态+因果W15 load分支。

current从所有输入路径排除；同一融合位置，同一初始化策略、预算与checkpoint选择规则。训练VAE的状态路径也要排除current，不能仅从推理token屏蔽。A0需要与新语义一致的训练，不能把现有P1直接当作无load/current基线。是否继承/切片原P1权重或统一重新初始化，需在修订合同明确后再冻结。原P1性能仅作为历史背景，不另增第四个主variant。

这样A2对A1才主要比较历史信息，而不是重复当前load或残留current；代价是要重建匹配的A0，不能声称输出与旧P1完全一致。

## 另一条合法但不同的问题

保留原P1的8D state在三组中完全一致，仅添加单帧/W15显式分支。其主问题应改成“在已有单帧load/current输入之上，显式时间编码是否有额外收益”。必须在revision写明current仍保留、A0并非无load；不能沿用current excluded的验收口径。

## 后续执行准备

澄清后建议沿用已有G0归一化h0夹爪L1作为主指标，明确Dev逐帧聚合与无ensemble/无relative clipping；物理单位MAE作次指标。若选择D2 episode等权物理h0 MAE，则须在任何训练前冻结并说明与G0的区别。这里只提议，不替用户冻结新主科研语义。

W15可采用episode首帧load左侧复制填充，A1/A2 load仅用Train36统计；两组使用相同小MLP输出维度和融合路径，记录参数量差别。全部属于待澄清后的实现方案，尚未执行/训练。Final12继续封存，不引入current分支、gate或残差控制。

Authorization: PENDING USER / PLANNER CLARIFICATION。原decision只读未修改。
