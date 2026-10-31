# P4.8 公开 benchmark 可行性与停止点交付

**RESULT_READY=true（G0–G1实证与G2阻断方案）；PILOT_READY=false；PROTOTYPE_SEMANTICS_BLOCKED。**

已完成：

- 独立venv、官方LeRobot0.4.4/固定commit、公开数据/仿真资产固定revision、依赖锁定；base包清单前后不变。pip check与实际eval --help通过。EGL调度库缺失、初始MuJoCo版本不兼容已在隔离目录修复，原错误保留。
- 真实EGL reset/step/render/成功标志通路通过；3条无策略smoke各20步，hard reset、同seed4701/init0的两相机/状态逐位差0。它们不是ACT/P1模型测试，不计算模型成功率。
- simulator task0按实际语言精确对应数据task24：alphabet soup→basket，44集/6867帧。全部episode单任务、长度/时间戳/帧序号正确；3集首/中/末9个真实样本与直接视频解码逐像素一致，动作100×7及末帧99padding没有串集。
- LIBERO状态8D=末端位置3+轴角3+双夹爪qpos2，动作7D=相对末端delta6+gripper，明确不是SO101同维数据。公开metadata10fps与实际物理控制20Hz差异已记录；未改时间戳、插值/重复动作或假定CLI fps改变物理步长。最终共用runtime与时基来源局限需要在G2配方审核中确认，未声称与论文结果数值可比。

**未启动模型训练/正式benchmark。** 公开数据缺少目标框/mask/目标patch标注，不能构造具有原版P1可追溯目标含义的4原型。decision第二阶段明确要求缺此类元数据时STOP并提交单独审核的确定性无监督方案。已交付prototype_spec.md及blocking_proposal.md：训练闭爪窗口的交互运动原型候选方案，以及4帧人工Train-only ROI的备选方案；均未执行，也未任意聚类、借用杯子原型或读取eval分割。

已记录用户并行要求：G2通过后，先执行合同内100步单/双路吞吐探针，再以实测效率安排两模型各10k；原定每模型10条配对初始状态可分给2–4个worker，保持总数、seed、hard-reset和预算不变。当前没有排队等待自动越过G2的训练任务。

CSV模型文件在无实验时仅保留列头，evidence_availability.json逐一说明；eval_rollouts.csv只有真实基础设施smoke记录。没有伪造checkpoint、训练吞吐、模型参数量或模型成功率。无Final15、无真机、无模型发布。独立Builder/Red-Team仍PENDING，自查不代替审核。

下一步供用户/网页规划助手审核：冻结原型来源/算法与共同runtime后，才能开始两模型pilot。本轮不自动进入全套10任务。

官方参考：[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)、[LeRobot集成](https://huggingface.co/docs/lerobot/libero)、[公开数据](https://huggingface.co/datasets/lerobot/libero)、[时基问题3368](https://github.com/huggingface/lerobot/issues/3368)。本地实测为20Hz；issue仅作旁证。
