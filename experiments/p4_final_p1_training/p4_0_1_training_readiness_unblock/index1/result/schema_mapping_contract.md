# 命名映射合同

目标state=['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos', 'gripper.load', 'gripper.curr']；action=['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']。实际current字段gripper.curr，gripper.current仅别名，重复别名报错。master只作遥操作元数据，不进自主策略。脚本scripts/p4_0_1_schema.py只返回新数组，不写原数据。全93391行已验证及列重排验证。生产dataset adapter接入仍待验证。相机top/gripper，原始RGB HWC640×480，模型应转换CHW float；本轮相机证据复用P4.0解码，未生产batch验证。
