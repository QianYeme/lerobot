# Normalization and timing

`legacy_fixed_preprocess_probe.json` 是本轮服务器隔离环境真实 Train episode 813 第0帧的参考数学计算。单次图像180°翻转后，Fixed 输入为 N(resize(RGB))；Legacy 为 N(resize(N(RGB)))；原型侧始终 N(resize(RGB))。Fixed 与原型侧逐元素差0，Legacy/Fixed 最大差 9.248488。所有张量有限。

**这不是 P1 运行时 hook，也不是 DINO/策略 forward 测试**。由于 G2 停止，没有构造 LIBERO bank 或两套模型，有效配置、初始化等价、CNN路径保持一致和实际输入 hook 均 NOT_RUN。

继承 G1：数据 timestamp/video metadata=10FPS，原生 MuJoCo 控制=20Hz、relative OSC_POSE、hard_reset=True；两路256×256 RGB，state8 / action7；gripper 为 action index6（而不是 SO101 index5）。state8不是7关节加夹爪。相机由官方 LiberoProcessor 翻转两轴一次。继承证据详见 inherited_index1/data_contract.md 与 sim_controller_contract.json。

未插帧、未重复动作、未改变控制周期或渲染器。本轮未运行模型闭环，故不能声称已解决时间基准的策略效果问题或复现了官方论文协议。后续正式训练/仿真须明确冻结10FPS训练时间索引与20Hz原生控制的对应协议；不能用增加预算掩盖时基错误。

没有使用全数据集 meta/stats.json 作为训练归一化；本轮未计算/使用任何策略训练统计。未来三个模型必须共用仅 Train35 统计。
