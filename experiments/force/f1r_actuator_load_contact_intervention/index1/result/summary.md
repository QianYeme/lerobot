# F1R：合同已读，等待真机环境与安全设置

F1R_COMPLETE = false  
SCIENTIFIC_OUTCOME = UNRESOLVED  
EVIDENCE_SCOPE = EXPLORATORY_SENSOR_CHARACTERIZATION

本轮需要同一固定臂姿态、同一安全夹爪闭合程序下5次FREE_CLOSE和5次CUP_CONTACT；旧数据或远程GPU不能替代新干预采集。

已提供PREPARATION_GUIDE.md及不可执行的protocol.json草案。按用户要求沿用旧采集指南的so101_follower、/dev/ttyACM0、robot.id=nn；历史leader为/dev/ttyACM1、id=nn，正式试验不使用。当前设备枚举/校准、固定姿态、开合范围与最大安全闭合、采样/程序/视频同步、急停与安全退出仍待现场确认。正式协议未冻结，有效试验0，未产生信号结论。

没有打开硬件或发送动作，没有policy训练、部署修改或自动收紧。待用户确认设备环境后继续工程工具及短安全设置；不猜测物理参数、不自行启动正式采集。

原F1/F1P/F1Q历史不改；D2/D2.1/G1未闭环与B3实际OC-ACT更正保留。
