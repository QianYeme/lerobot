# Timing / control audit

**TIMEBASE_BLOCKED（严格的采集物理时基来源仍未证实）**；独立的原生接口工程核验为PASS。不能把两个层次混成一个通过。

本轮实际reset/step：原生relative OSC_POSE，20Hz；一次dummy action的仿真时间增量0.050000000000000044s。没有重复动作、插值或改control_freq。state8为EEF位置3+轴角3+夹爪qpos2，action7为相对动作6+夹爪1；无SO101角度/load/current。实际源文件SHA见timing_probe.json。此处使用G1已有init0/seed4701作工程检查，无图像提供给原型生成，没有执行训练策略闭环。

数据timestamp/video=10FPS。ACT工程smoke使用delta_timestamps=i/10，即每个未来索引增加一条原始动作，chunk100涵盖0–9.9秒的**编码时间**；没有插值或跨episode动作，padding保持原生LeRobotDataset行为。未来若采用官方原生推理，每次env.step消耗一个动作，100步约5秒**仿真时间**，n_action_steps默认100、无temporal ensemble。这里明确披露两个跨度，不声称物理等价。

核查了[官方LIBERO指南](https://huggingface.co/docs/lerobot/libero)、[固定v0.4.4环境源代码](https://github.com/huggingface/lerobot/blob/8fff0fde7c79f23a93d845d1a50e985de01f8b8a/src/lerobot/envs/libero.py)、[上游频率问题3368](https://github.com/huggingface/lerobot/issues/3368)。官方通路确认原生单动作/单step以及20Hz环境；修改配置fps不会自动重设物理控制时钟。

也追查了[原LeRobot格式来源卡](https://huggingface.co/datasets/physical-intelligence/libero)，其指向[OpenVLA modified RLDS](https://huggingface.co/datasets/openvla/modified_libero_rlds)。这些公开说明不足以在本轮证明固定revision的逐行动作在转换前是否抽帧、原始采集物理周期与10FPS时间戳完全相等。没有以相同总帧数代替逐步物理时基证据。

因此本轮不宣布正式闭环时基已对齐。下一决策须批准以官方原生“序号→单step20Hz”为所报告协议并披露此局限，或补充固定数据来源转换证据。无需、也不能凭猜测把动作重复两次或把控制器改10Hz。
