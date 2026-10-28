# P1接口审查

P1_8D_6D_COMPATIBLE = UNKNOWN（结构字段可映射，原始张量不直接匹配；单位/标定未证实）
原始state顺序：shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos, gripper.load, gripper.curr, master_gripper.pos。
动作顺序为前六项，6D。历史f3_runtime.py的M0路径明确使用六位置+load/curr共8D，主臂不在输入中。不是删除P1原load/current。
后续建议准备只去掉master的独立派生版本；本轮未执行。不能把9D直接输入8D模型。
机器人代码支持degrees或range归一化；仅凭robot_type和数值范围不能证明采集端实际设置/标定一致。动作/状态同名不单独证明绝对位置约定。需采集配置证据。
相机声明及解码640×480、30FPS，top俯视、gripper腕部；未更改相机映射。
