# D2 动作链诊断交付（部分完成，2026-10-14）

DIAGNOSIS_COMPLETE=FALSE；diagnostic_category=UNRESOLVED；acceptance_status=PARTIAL_PASS。

已执行元数据与 split 审计：36 Train / 12 Dev，互不重叠；声明 state 8D、action 6D、30 FPS。未读取 Final episode 内容。实际单位、时序对齐与模型输入仍需完整审计，不能宣称 schema Gate PASS。

代码审计：正时间集成系数优先较早预测，集成启用时每步预测完整 chunk；关闭时按 n_action_steps 消耗队列。限幅在 follower.send_action 中对新读取 Present_Position 生效，录制脚本保存 sent_action。历史录制代码身份、实际部署解析参数均未核验。

## 实际完成

服务器 RTX5090 上 P1、R3 各完成12 Dev、5748帧的teacher-forced推理和四配置回放，成功运行耗时分别138.26/117.30秒。P1模型/processor文件哈希见p1_unphased_metrics.json，R3见r3_unphased_metrics.json。48个Train/Dev信号审计覆盖17243/5748帧，数值有限、帧号连续；时间戳步长误差最大约5.1e-7秒。没有读取Final行、训练或真机执行。

P1实际时间权重与安全限幅函数校验通过，见replay_validation.json；重建了同episode未来动作目标，按尾部mask报告合法full-chunk误差。未完成normalized空间集成与物理空间集成的独立等价验证，不将此项视为PASS。

## 初步行为证据

未分阶段、反归一化数据单位下：P1原始h0机械臂MAE=2.198、夹爪=1.477；R3=3.392/2.720。它们是专家观测上的输出误差，不是真机性能排名。

P1回放机械臂/夹爪MAE：off+5为1.911/1.568，off+2为1.708/1.673；0.01+5为2.732/3.140，0.01+2为2.090/2.273。同限幅下时间集成增加全任务误差，限幅2也明显影响输出。但阶段unknown，不能归因到低抬升和短保持；限幅降低teacher-forced误差不能推出真机更好。

## 关键阻塞与反证

实际G1 B3 checkpoint是model_type=oc_act、object_conditioning_enabled=true，原T3 manifest也冻结OC-ACT branch；与本D2合同的标准ACT定义冲突。未替换模型、未关闭模块，B3推理暂停。网页GPT需先澄清身份；不能自行选B0代替。

没有Formal4经人工审核视觉事件标签，尚未完成48集双相机视觉事件审查。demonstration_events.csv对240个事件如实记UNKNOWN；阶段覆盖为0，不将信号proxy冒充离桌/接触桌面真值。实际运行参数、单位标定、延迟/可达性、prototype哈希仍不充分；没有真实运行日志。旧失败日志保存于服务器。

因此G0/G1/G2/G5/G6仅部分满足，G3/G7为NA；不选择唯一根因，不宣称D2_COMPLETE或G1正式成功。

## 单一下一步建议

先补齐合法诊断证据：由网页GPT澄清B3身份合同，并安排可追溯Formal4视觉事件审核；在此之前不启动模型优化训练。对低抬升/短保持的DATA、PREDICTION、POSTPROCESS、CLOSED_LOOP解释仍保留竞争。此result为部分交付，不等于用户要求的完整D2已完成。

