# F1 信号与因果时序审计

证据为当前 det HEAD，文件哈希见 provenance.json；历史 Formal4 采集时源码/固件/标定原件尚未独立核验。因此下述是当前实现语义，不冒充完整历史部署等价。

| 时刻 | 源码证据 | 允许的预测输入 |
|---|---|---|
| t观测 | so_follower.py:get_observation，先读五关节，再一次sync_read_multi读夹爪position/load/current，随后取相机最新帧 | obs≤t；不同设备并非严格同时曝光 |
| t生成命令 | lerobot_record.py:348先get_observation，之后teleop/policy生成动作 | 同一行action[t]尚不已知，禁止使用 |
| t发送 | lerobot_record.py:417调用send_action；follower可能依据fresh位置限幅后写Goal_Position | 录制action[t]是返回的发送目标，非原始leader目标、非执行完成反馈 |
| t记录 | lerobot_record.py:421–423合并此前observation与sent_action | 下一轮可使用action[t]；本轮最多action[t−1] |

S0通道固定为pos[t]、pos[t]−pos[t−1]、action[t−1]、action[t−1]−action[t−2]。每通道固定当前值/窗口mean/std(ddof=0)/末减首；S1/S_current/S2仅增加指定信号的相同模板。W=1/9/15，不跨episode、不填充；最早合法目标帧分别为2/10/16。各probe统一采用全部24列有限性掩码，跨窗口比较需逐episode取合法帧交集。

## 字段、符号与单位

- state顺序为五关节pos、gripper.pos、gripper.load、gripper.curr；action第5索引为gripper.pos。元数据8D/6D、声明30FPS；允许48集帧号连续，timestamp≈frame/30验证通过。
- gripper.pos：MotorNormMode.RANGE_0_100，motors_bus.py:_normalize按校准范围限界后映射0–100。它不是毫米张口，标定方向与实际开合需视觉确认；不能仅按数字方向生成接触标签。
- load/current：Feetech STS3215分别读Present_Load地址60/2字节、Present_Current地址69/2字节；tables.py二者sign-magnitude符号位10；feetech.py:sync_read_multi逐字段解码。符号是编码方向，不足以定义夹持/拉力正负。
- NORMALIZED_DATA仅位置相关字段；load/current保留有符号寄存器计数。当前路径未见额外物理单位换算或软件限幅。不声称牛顿、安培、已标定握力。
- configure中的Max_Torque_Limit=500、Protection_Current=250、Overload_Torque=25是控制/保护设置；不能直接当作读数饱和上下限。硬件饱和及固件滤波/延迟UNKNOWN。
- 同步总线块读取减少夹爪字段间通信差异，但没有硬件时间戳证明真实传感/相机延迟。视频索引沿用D2.1的30FPS元数据offset及完整解码记录，尚未用独立时钟验证曝光与遥测同步。
- 历史输入标准化、具体设备标定与采集源码版本未完全核验；只使用存储的原生字段进行诊断，scaler尚未拟合。训练集stats不作为全数据归一化依据。

## 当前结论

可构造保守、无未来命令的历史特征，但完整真实部署时序仍有上述局限。这个工程结论不证明load/current可识别接触。
