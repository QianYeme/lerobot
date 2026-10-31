# Input / scope audit

- 延用冻结Train35/Dev9，split SHA由r2保留且不改。Train35的23集用作候选发现、12集按每第三个索引固定保留作无标签proxy检查；这些不是新增Dev集，不改变策略训练分割。
- 模型特征、事件和原型均只使用Train35目标帧。共享Parquet/MP4文件可包含其它episode；表格读取后立即筛选Train35，视频顺序解码时只将目标Train帧保存、送入DINO。没有将“下载共享分片”错误描述为所有文件只含训练集。
- 自动构建只使用Top RGB、action[6]闭合命令和公开EEF/gripper state；闭合命令不是接触标签。没有人工中心、ROI、Mask，未复用杯bank/历史四坐标；没有仿真分割/物体位姿/目标类别输入。
- 静态背景和premover仅为运动proxy，不能冒充GT。独立Train proxy只有7个正例patch，语义对应仍未知。时序图仅在选取/冻结gate结果之后供诊断查看，没有据此选新帧/修改bank/改变阈值。
- 没有Dev9模型评估、没有读取SO101 Final15、没有新机器人数据、没有实际机械臂操作、没有P4.7修改或HF/Git发布。
- source DINO/ResNet使用已存在公开预训练缓存；ACT的state/action统计只从Train35的5476帧计算，未使用全库meta/stats.json。
